#!/usr/bin/env python3
"""
SW-emulated FL node — host-side process mimicking a FireSim FPGA client.

Uses FlexDNN (from fl-platform/numpy_version) so architecture is configurable
via --arch.  Emits the same protocol as firesim_client.py:
    FL_ARCH:<descriptor>
    FL_TASK:<base64 float32[16]>
    FL_WEIGHTS_B64:<base64 float32[N]>

After each round it blocks until the manager writes a fresh global_model.npy
(or personalized_model.npy) to --out-dir, then loads it and continues.

Usage:
    python3 sw_emu_node.py --emu-id 0 --out-dir <path> --arch 3,16,1
    python3 sw_emu_node.py --emu-id 1 --out-dir <path> --arch 3,64,32,8,1
"""

import argparse
import base64
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

FL_PLATFORM = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, FL_PLATFORM)
from Model.DNNModels import FlexDNN, make_task_embedding  # noqa: E402
from Updater.SGD import SGD                               # noqa: E402

_running = True


def _handle_signal(sig, frame):
    global _running
    _running = False


def make_dataset(emu_id: int):
    rng = np.random.RandomState(42 + emu_id * 997)
    X   = rng.randn(200, 3)
    w   = rng.randn(3, 1)
    y   = X @ w + 0.1 * rng.randn(200, 1)
    return X.astype(np.float64), y.astype(np.float64)


def train_round(model, updater, X, y, epochs: int, batch: int, lr: float) -> float:
    n = len(X)
    last_loss = 0.0
    for _ in range(epochs):
        idx = np.random.permutation(n)
        Xs, ys = X[idx], y[idx]
        losses = []
        for start in range(0, n, batch):
            xb = Xs[start:start + batch]
            yb = ys[start:start + batch]
            grad = updater.Update(model, xb, yb)
            model.para = model.para - lr * grad
            model.para = model.ProximalOperator(model.para, 0.0)
            losses.append(model.ComputeLoss(model.para, xb, yb))
        last_loss = float(np.mean(losses))
    return last_loss


def wait_for_round_signal(signal_path: Path, expected_round: int, label: str) -> bool:
    """Block until manager writes round_ready.txt with value >= expected_round."""
    print(f"[{label}] waiting for updated model from manager...", flush=True)
    while _running:
        if signal_path.exists():
            try:
                if int(signal_path.read_text().strip()) >= expected_round:
                    return True
            except (ValueError, OSError):
                pass
        time.sleep(0.3)
    return False


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    parser = argparse.ArgumentParser(description="SW-EMU FL node (FlexDNN)")
    parser.add_argument("--emu-id",       type=int,   required=True)
    parser.add_argument("--out-dir",      required=True)
    parser.add_argument("--arch",         type=str,   default=None,
                        help="Comma-separated layer sizes, e.g. 3,16,1")
    parser.add_argument("--round-epochs", type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--global-model", default=None)
    parser.add_argument("--delay",        type=float, default=0.0)
    args = parser.parse_args()

    label   = f"SW-EMU{args.emu_id}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gm_path = out_dir / "global_model.npy"

    if args.delay > 0:
        print(f"[{label}] sleeping {args.delay:.0f}s (simulated boot)...", flush=True)
        time.sleep(args.delay)

    # ── Build model ──────────────────────────────────────────────────────────
    if args.arch is not None:
        arch = [int(x) for x in args.arch.split(",")]
    else:
        arch = [3, 16, 1]   # ShallowDNN default

    model    = FlexDNN(arch)
    arch_str = model.arch_descriptor
    n_params = model.NumParameters()

    # ── Banner ───────────────────────────────────────────────────────────────
    arch_arrow = " → ".join(str(s) for s in arch)
    print("", flush=True)
    print("╔════════════════════════════════════════════════════════════════╗", flush=True)
    print("║          FedML Client (SW Emulation — Host Process)            ║", flush=True)
    print("╚════════════════════════════════════════════════════════════════╝", flush=True)
    print("", flush=True)
    print(f"  EMU ID         : {args.emu_id}  ({label})", flush=True)
    print(f"  Architecture   : {arch_arrow}", flush=True)
    print(f"  Parameters     : {n_params}", flush=True)
    print(f"  Round Epochs   : {args.round_epochs}", flush=True)
    print(f"  Batch Size     : {args.batch}", flush=True)
    print(f"  Learning Rate  : {args.lr}", flush=True)
    print(f"  Dataset Seed   : {42 + args.emu_id * 997}", flush=True)
    print("", flush=True)
    print("  Protocol: FL_ARCH → FL_TASK → FL_WEIGHTS_B64 per round", flush=True)
    print("", flush=True)

    # ── Optionally warm-start from a supplied global model ───────────────────
    gm_src = Path(args.global_model) if args.global_model else gm_path
    if gm_src.exists():
        flat = np.load(gm_src).flatten()
        if flat.size == n_params:
            model.para = flat.reshape(-1, 1).astype(np.float64)
            model._unpack_para()
            print(f"[{label}] loaded warm-start model shape={flat.shape}", flush=True)
        else:
            print(f"[{label}] model size mismatch ({flat.size} vs {n_params}) — random init",
                  flush=True)
    else:
        print(f"[{label}] no warm-start model — random init", flush=True)

    X, y    = make_dataset(args.emu_id)
    updater = SGD(n_params, X, y)

    # ── Task embedding (once, from local data statistics) ────────────────────
    task_vec = make_task_embedding(X, y, args.emu_id)
    task_b64 = base64.b64encode(task_vec.tobytes()).decode("ascii")

    print(f"[{label}] FL_CLIENT_START emu_id={args.emu_id} "
          f"arch={arch} round_epochs={args.round_epochs}", flush=True)

    signal_path = out_dir / "round_ready.txt"

    round_num = 0
    while _running:
        round_num += 1
        print(f"[{label}] FL_ROUND_START round={round_num}", flush=True)

        loss = train_round(model, updater, X, y,
                           args.round_epochs, args.batch, args.lr)

        flat  = model.para.flatten().astype(np.float32)
        w_b64 = base64.b64encode(flat.tobytes()).decode("ascii")

        # Protocol lines parsed by fedml_manager
        print(f"FL_ARCH:{arch_str}",     flush=True)
        print(f"FL_TASK:{task_b64}",     flush=True)
        print(f"FL_LOSS:{loss:.6f}",     flush=True)
        print(f"FL_WEIGHTS_B64:{w_b64}", flush=True)

        # Block until manager writes round_ready.txt with this round number.
        # Uses a plain counter file — immune to filesystem mtime resolution races.
        if not wait_for_round_signal(signal_path, round_num, label):
            break

        flat = np.load(gm_path).flatten()
        if flat.size == n_params:
            model.para = flat.reshape(-1, 1).astype(np.float64)
            model._unpack_para()
            print(f"[{label}] loaded personalized model for round={round_num + 1} "
                  f"shape={flat.shape}", flush=True)
        else:
            print(f"[{label}] model size mismatch on reload ({flat.size} vs {n_params})"
                  f" — keeping current params", flush=True)

    print(f"[{label}] FL_CLIENT_EXIT rounds_completed={round_num}", flush=True)


if __name__ == "__main__":
    main()
