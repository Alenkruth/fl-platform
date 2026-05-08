#!/usr/bin/env python3
"""
Host-side Federated Learning manager for FireSim.

Pure collector — does not call any FireSim CLI commands. Run FireSim yourself
(infrasetup + runworkload) in a separate terminal, then start this manager to
connect into the live simulation.

  FPGA nodes  — tails the uartlog written by FireSim in real-time; decodes
                FL_ARCH, FL_TASK, and FL_WEIGHTS_B64 lines per node.
  SW-EMU nodes — launches host-side subprocess nodes; reads their stdout.

After collecting one weight vector per node the manager runs the TAHN bridge
function (FedMBridge-style aggregation), saves the personalized models, and
unblocks SW-EMU nodes for the next round. Runs until Ctrl+C.

Architecture protocol (printed by firesim_client.py each round):
    FL_ARCH:<descriptor>          e.g. FL_ARCH:3,16,1
    FL_TASK:<base64 float32[16]>  data-statistics task embedding
    FL_WEIGHTS_B64:<base64>       flat float32 weight vector

If FL_ARCH / FL_TASK are absent (older client images) the manager falls back
to DEFAULT_ARCH and null task embeddings.

Usage:
    # FPGA only (auto-detects latest results-workload/ directory):
    python3 fedml_manager.py --fpga-nodes 1

    # Two FPGAs with explicit uartlog paths:
    python3 fedml_manager.py --fpga-nodes 2 \\
        --uartlogs /data/akrish/firesim/simulation/sim_slot_0/uartlog \\
                   /data/akrish/firesim/simulation-1/sim_slot_0/uartlog

    # Resume from a previous global model:
    python3 fedml_manager.py --fpga-nodes 1 --resume results-workload/round_005_global.npy
"""

import argparse
import base64
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tahn import TAHN

# ── Paths / config ────────────────────────────────────────────────────────────
SW_EMU_SCRIPT   = Path(__file__).parent / "sw_emu_node.py"
WORKLOAD_NAME   = "ubuntu-fedml"
RESULTS_BASE    = Path("/data/akrish/scratch/fedml/results")
ROUND_EPOCHS    = 256
SW_EMU_DELAY    = 0    # no simulated boot delay for SW-EMU
DEFAULT_SIM_DIR = Path("/data/akrish/firesim/simulation")

# Fallback architectures when FL_ARCH is not received (legacy FPGA client images).
DEFAULT_ARCHS = {
    "FPGA-NODE0": [3, 16, 1],
    "FPGA-NODE1": [3, 16, 1],
}
DEFAULT_ARCH_FALLBACK = [3, 16, 1]

# Per-emu-id architecture for SW-EMU nodes (heterogeneous by design).
SW_EMU_ARCHS = {
    0: [3, 8, 1],           #  41 params — tiny shallow net
    1: [3, 32, 16, 1],      # 673 params — wide-deep net
    2: [3, 16, 8, 1],       # 209 params — DeepDNN
    3: [3, 64, 1],          # 321 params — very wide shallow net
}

_shutdown    = threading.Event()
_print_lock  = threading.Lock()
_meta_lock   = threading.Lock()
_node_meta: dict = {}   # {label: {'arch': [...], 'task': np.ndarray|None}}


def mprint(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _handle_signal(sig, frame):
    mprint("\n[manager] interrupt — shutting down", flush=True)
    _shutdown.set()


def _set_node_arch(label: str, arch: list):
    with _meta_lock:
        _node_meta.setdefault(label, {})['arch'] = arch


def _set_node_task(label: str, task_vec: np.ndarray):
    with _meta_lock:
        _node_meta.setdefault(label, {})['task'] = task_vec


def _get_node_meta(label: str) -> dict:
    with _meta_lock:
        return dict(_node_meta.get(label, {}))


# ── Weight helpers ────────────────────────────────────────────────────────────

def decode_weights(b64_str: str) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    if len(raw) % 4 != 0:
        raise ValueError(f"base64 payload length {len(raw)} not a multiple of 4 bytes")
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    if arr.size == 0:
        raise ValueError("decoded weight array is empty")
    return arr


# ── FPGA uartlog tail thread ──────────────────────────────────────────────────

def tail_fpga_uartlog(log_path: Path, label: str, weight_q: queue.Queue):
    """Wait for uartlog to appear, tail it, decode FL protocol lines.
    Manager drives all console output — nothing is echoed from the uartlog."""
    mprint(f"[manager] {label}: waiting for uartlog: {log_path}", flush=True)
    while not log_path.exists() and not _shutdown.is_set():
        time.sleep(1)
    if _shutdown.is_set():
        return
    mprint(f"[manager] {label}: tailing {log_path}", flush=True)
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(0, 2)  # start from end — only process lines written after manager starts
            pending_b64 = ""
            pending_loss = float("nan")
            while not _shutdown.is_set():
                line = f.readline()
                if not line:
                    time.sleep(0.1)
                    continue
                line = line.rstrip()

                if "FL_ARCH:" in line:
                    arch_str = line.split("FL_ARCH:", 1)[1].strip()
                    try:
                        _set_node_arch(label, [int(x) for x in arch_str.split(",")])
                    except ValueError:
                        pass

                elif "FL_TASK:" in line:
                    try:
                        task_vec = np.frombuffer(
                            base64.b64decode(line.split("FL_TASK:", 1)[1].strip()),
                            dtype=np.float32).copy()
                        _set_node_task(label, task_vec)
                    except Exception:
                        pass

                elif "FL_LOSS:" in line:
                    try:
                        pending_loss = float(line.split("FL_LOSS:", 1)[1].strip())
                    except ValueError:
                        pass

                elif "FL_ROUND_DONE" in line and "loss=" in line:
                    try:
                        pending_loss = float(line.split("loss=", 1)[1].strip())
                    except ValueError:
                        pass

                elif "FL_WEIGHTS_B64:" in line:
                    pending_b64 = line.split("FL_WEIGHTS_B64:", 1)[1].strip()
                    try:
                        w = decode_weights(pending_b64)
                        while not weight_q.empty():
                            try: weight_q.get_nowait()
                            except queue.Empty: break
                        weight_q.put((w, pending_loss))
                        pending_b64 = ""
                        pending_loss = float("nan")
                    except Exception:
                        pass  # incomplete line — accumulate

                elif pending_b64:
                    pending_b64 += line.strip()
                    try:
                        w = decode_weights(pending_b64)
                        while not weight_q.empty():
                            try: weight_q.get_nowait()
                            except queue.Empty: break
                        weight_q.put((w, pending_loss))
                        pending_b64 = ""
                        pending_loss = float("nan")
                    except Exception:
                        pass  # still incomplete

    except Exception as e:
        mprint(f"[manager] {label}: tail error: {e}", flush=True)


# ── SW-EMU stream thread ──────────────────────────────────────────────────────

def stream_sw_emu(proc: subprocess.Popen, label: str, weight_q: queue.Queue):
    """Parse SW-EMU stdout for FL protocol lines. Manager drives all console output."""
    pending_loss: float = float("nan")
    for raw in proc.stdout:
        if _shutdown.is_set():
            break
        line = raw.rstrip()
        if "FL_ARCH:" in line:
            arch_str = line.split("FL_ARCH:", 1)[1].strip()
            try:
                _set_node_arch(label, [int(x) for x in arch_str.split(",")])
            except ValueError:
                pass
        elif "FL_TASK:" in line:
            b64 = line.split("FL_TASK:", 1)[1].strip()
            try:
                _set_node_task(label,
                    np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy())
            except Exception:
                pass
        elif "FL_LOSS:" in line:
            try:
                pending_loss = float(line.split("FL_LOSS:", 1)[1].strip())
            except ValueError:
                pass
        elif "FL_WEIGHTS_B64:" in line:
            b64 = line.split("FL_WEIGHTS_B64:", 1)[1].strip()
            try:
                weight_q.put((decode_weights(b64), pending_loss))
                pending_loss = float("nan")
            except Exception as e:
                mprint(f"[manager] {label}: decode error: {e}", flush=True)


# ── SW-EMU node launcher ──────────────────────────────────────────────────────

def _arch_n_params(arch: list) -> int:
    return sum(arch[i] * arch[i + 1] + arch[i + 1] for i in range(len(arch) - 1))


def launch_sw_emu(emu_id: int, out_dir: Path, dry_run: bool) -> subprocess.Popen | None:
    arch = SW_EMU_ARCHS.get(emu_id, DEFAULT_ARCH_FALLBACK)
    cmd = [
        sys.executable, str(SW_EMU_SCRIPT),
        "--emu-id",       str(emu_id),
        "--out-dir",      str(out_dir),
        "--arch",         ",".join(str(s) for s in arch),
        "--round-epochs", str(ROUND_EPOCHS),
        "--delay",        str(SW_EMU_DELAY),
    ]
    label = f"SW-EMU{emu_id}"
    if dry_run:
        mprint(f"[manager] {label}: dry-run, not started", flush=True)
        return None
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    parser = argparse.ArgumentParser(description="FireSim Federated ML manager")
    parser.add_argument("--fpga-nodes",   type=int, default=1,
                        help="Number of FPGA nodes (default 1)")
    parser.add_argument("--sw-emu-nodes", type=int, default=0, metavar="M",
                        help="Number of SW-emulated host-side nodes (default 0)")
    parser.add_argument("--sw-emu-archs", type=str, nargs="+", default=None,
                        metavar="ARCH",
                        help="Architecture per SW-EMU node as comma-separated sizes, "
                             "e.g. 3,16,1 3,32,16,1  (overrides built-in SW_EMU_ARCHS)")
    parser.add_argument("--fpga-archs",   type=str, nargs="+", default=None,
                        metavar="ARCH",
                        help="Architecture per FPGA node as comma-separated sizes, "
                             "e.g. 3,16,1 3,16,1  (overrides built-in DEFAULT_ARCHS)")
    parser.add_argument("--sim-dir",      type=str, default=str(DEFAULT_SIM_DIR),
                        help=f"FireSim sim dir containing sim_slot_N/ "
                             f"(default: {DEFAULT_SIM_DIR})")
    parser.add_argument("--uartlogs",     type=str, nargs="+", default=None,
                        metavar="PATH",
                        help="Explicit uartlog path(s), one per FPGA node. "
                             "Overrides --sim-dir.")
    parser.add_argument("--resume",       type=str, default=None,
                        help="Path to global_model.npy to warm-start from")
    parser.add_argument("--rounds",        type=int, default=0,
                        help="Stop after N rounds (0 = run until Ctrl+C)")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print SW-EMU launch commands without starting them")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)

    if args.sw_emu_archs is not None:
        for i, spec in enumerate(args.sw_emu_archs):
            SW_EMU_ARCHS[i] = [int(x) for x in spec.split(",")]

    if args.fpga_archs is not None:
        for i, spec in enumerate(args.fpga_archs):
            DEFAULT_ARCHS[f"FPGA-NODE{i}"] = [int(x) for x in spec.split(",")]

    n_fpga   = args.fpga_nodes
    n_sw_emu = args.sw_emu_nodes
    n_total  = n_fpga + n_sw_emu

    global_model = None
    if args.resume:
        global_model = np.load(args.resume).flatten().astype(np.float32)

    # Build per-FPGA uartlog paths
    sim_dir = Path(args.sim_dir)
    if args.uartlogs is not None:
        if len(args.uartlogs) != n_fpga:
            parser.error(
                f"--uartlogs: got {len(args.uartlogs)} paths but --fpga-nodes={n_fpga}")
        fpga_log_paths = [Path(p) for p in args.uartlogs]
    else:
        fpga_log_paths = [sim_dir / f"sim_slot_{i}" / "uartlog"
                          for i in range(n_fpga)]

    # ── Print client roster before launching ──────────────────────────────────
    mprint(f"[manager] {n_fpga} FPGA + {n_sw_emu} SW-EMU clients:", flush=True)
    for i in range(n_fpga):
        arch = DEFAULT_ARCHS.get(f"FPGA-NODE{i}", DEFAULT_ARCH_FALLBACK)
        mprint(f"  FPGA-NODE{i}  arch={arch}  uartlog={fpga_log_paths[i]}", flush=True)
    for emu_id in range(n_sw_emu):
        arch = SW_EMU_ARCHS.get(emu_id, DEFAULT_ARCH_FALLBACK)
        mprint(f"  SW-EMU{emu_id}    arch={arch}", flush=True)
    if args.resume:
        mprint(f"[manager] warm-start: {args.resume}", flush=True)

    # ── TAHN (init before node setup so seed_prev_gen can be called during launch)
    tahn = TAHN(seed=42)

    # ── Per-node weight queues ─────────────────────────────────────────────────
    weight_queues: dict[str, queue.Queue] = {}
    threads = []

    for i in range(n_fpga):
        label  = f"FPGA-NODE{i}"
        arch   = DEFAULT_ARCHS.get(label, DEFAULT_ARCH_FALLBACK)
        q      = queue.Queue()
        weight_queues[label] = q
        _set_node_arch(label, arch)
        # FPGA has no write-back path — seed TAHN and log, skip file I/O
        init_w = (global_model[:_arch_n_params(arch)]
                  if global_model is not None else
                  np.zeros(_arch_n_params(arch), dtype=np.float32))
        tahn.seed_prev_gen(label, arch, init_w)
        mprint(f"[manager] {label}: initial model dispatched  n_params={_arch_n_params(arch)}"
               f"  (FPGA: log only, no write-back)", flush=True)
        t = threading.Thread(target=tail_fpga_uartlog,
                             args=(fpga_log_paths[i], label, q), daemon=True)
        t.start()
        threads.append(t)

    sw_emu_procs = []
    for emu_id in range(n_sw_emu):
        label    = f"SW-EMU{emu_id}"
        out_dir  = RESULTS_BASE / f"sw-emu{emu_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        arch     = SW_EMU_ARCHS.get(emu_id, DEFAULT_ARCH_FALLBACK)
        n_params = _arch_n_params(arch)

        # Write initial model before launching so client always warm-starts from
        # a server-provided model, not random init.  Use --resume model if given,
        # otherwise small random init scaled to avoid saturation.
        init_w = global_model[:n_params] if (global_model is not None
                                             and global_model.size >= n_params) \
                 else (np.random.randn(n_params).astype(np.float32) * 0.1)
        np.save(out_dir / "global_model.npy", init_w)
        (out_dir / "round_ready.txt").write_text("0")
        tahn.seed_prev_gen(label, arch, init_w)
        mprint(f"[manager] {label}: initial model dispatched  n_params={n_params}", flush=True)

        q = queue.Queue()
        weight_queues[label] = q
        _set_node_arch(label, arch)
        proc = launch_sw_emu(emu_id, out_dir, args.dry_run)
        sw_emu_procs.append(proc)
        if proc is not None:
            t = threading.Thread(target=stream_sw_emu,
                                 args=(proc, label, q), daemon=True)
            t.start()
            threads.append(t)

    # ── Aggregation loop ───────────────────────────────────────────────────────
    all_labels = list(weight_queues.keys())
    max_rounds = args.rounds
    mprint(f"[manager] {'running for ' + str(max_rounds) + ' rounds' if max_rounds else 'running until Ctrl+C'}",
           flush=True)

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    round_num = 0
    try:
        while not _shutdown.is_set():
            round_num += 1
            mprint(f"\n[manager] ===== round {round_num} =====", flush=True)

            # Collect one weight vector per node — poll all queues concurrently
            # so a slow FPGA node doesn't block collection from faster nodes.
            pending = set(all_labels)
            weights: dict[str, np.ndarray] = {}
            losses:  dict[str, float]      = {}
            while pending and not _shutdown.is_set():
                for label in list(pending):
                    try:
                        w, loss = weight_queues[label].get_nowait()
                        weights[label] = w
                        losses[label]  = loss
                        pending.discard(label)
                    except queue.Empty:
                        pass
                if pending:
                    time.sleep(0.1)
            for label in all_labels:
                if label not in weights:
                    continue
                w, loss = weights[label], losses[label]
                meta = _get_node_meta(label)
                loss_str = f"  loss={loss:.6f}" if not np.isnan(loss) else ""
                mprint(f"[manager] {label}: weights received  "
                       f"arch={meta.get('arch', '?')}  n_params={w.size}{loss_str}", flush=True)

            if not weights:
                break

            # ── Build TAHN client dict ─────────────────────────────────────────
            clients = {}
            for label, w in weights.items():
                meta = _get_node_meta(label)
                clients[label] = {
                    'arch':      meta.get('arch', DEFAULT_ARCH_FALLBACK),
                    'task_init': meta.get('task', None),
                    'weights':   w,
                }

            # ── TAHN bridge aggregation ────────────────────────────────────────
            personalized = tahn.aggregate(clients)
            mprint(tahn.round_log(), flush=True)

            # Save personalized model for each client
            for label, pers_w in personalized.items():
                out_path = RESULTS_BASE / f"round_{round_num:03d}_{label}_personalized.npy"
                np.save(out_path, pers_w)

            # Also save a flat-averaged global for backward compatibility
            all_w = list(weights.values())
            # Align sizes for simple mean (use shortest common subset)
            min_len = min(w.size for w in all_w)
            global_agg = np.mean([w[:min_len] for w in all_w], axis=0)
            global_path = RESULTS_BASE / f"round_{round_num:03d}_global.npy"
            np.save(global_path, global_agg)

            # Dispatch personalized models — SW-EMU receives file, FPGA log only
            for label, pers_w in personalized.items():
                mprint(f"[manager] {label}: model dispatched  n_params={pers_w.size}", flush=True)
                if label.startswith("SW-EMU"):
                    emu_id   = int(label.split("SW-EMU")[1])
                    node_dir = RESULTS_BASE / f"sw-emu{emu_id}"
                    np.save(node_dir / "global_model.npy", pers_w)
                    (node_dir / "round_ready.txt").write_text(str(round_num))
                # FPGA nodes: personalized weight computed and logged, write-back skipped

            if max_rounds and round_num >= max_rounds:
                mprint(f"\n[manager] reached {max_rounds} rounds — stopping", flush=True)
                _shutdown.set()
                break

    except KeyboardInterrupt:
        _shutdown.set()

    # ── Cleanup ────────────────────────────────────────────────────────────────
    mprint(f"\n[manager] completed {round_num} round(s) — cleaning up", flush=True)
    for proc in sw_emu_procs:
        if proc is not None:
            proc.terminate()
    for proc in sw_emu_procs:
        if proc is not None:
            proc.wait()
    mprint(f"[manager] done. Results in {RESULTS_BASE}/", flush=True)


if __name__ == "__main__":
    main()
