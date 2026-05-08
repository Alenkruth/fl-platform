#!/usr/bin/env python3
"""
FireSim FL client — runs on the FPGA node inside the simulation.

Architecture is selected by node_id:
    node 0  →  FlexDNN([3, 16, 1])      (ShallowDNN, 81 params)
    node 1+ →  FlexDNN([3, 16, 8, 1])   (DeepDNN,   209 params)

After each local training round the client prints three lines to stdout,
which the host manager captures via the uartlog:
    FL_ARCH:<descriptor>         e.g. FL_ARCH:3,16,1
    FL_TASK:<base64 float32[16]> data-statistics task embedding
    FL_WEIGHTS_B64:<base64>      flat float32 weight vector
    [label] FL_ROUND_DONE round=N loss=X

Node ID is read from /var/firesim-node-id and used to seed a unique local
dataset, simulating non-IID data across edge devices.

Usage (inside simulation):
    python3.10 /root/fl-platform/numpy_version/firesim_client.py
    python3.10 /root/fl-platform/numpy_version/firesim_client.py --round-epochs 64
"""

import argparse
import base64
import os
import signal
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Model.DNNModels import FlexDNN, make_task_embedding
from Updater.SGD import SGD

WEIGHTS_PATH  = "/root/local_weights.npy"
NODE_ID_PATH  = "/var/firesim-node-id"

# Architecture per node ID: extend the list for more nodes.
NODE_ARCHS = {
    0: [3, 16, 1],       # ShallowDNN — 81 params
    1: [3, 16, 8, 1],    # DeepDNN   — 209 params
}
DEFAULT_ARCH = [3, 16, 8, 1]   # fallback for node_id >= 2

_running = True


def _handle_signal(sig, frame):
    global _running
    _running = False


def get_node_id() -> int:
    try:
        with open(NODE_ID_PATH) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def make_dataset(node_id: int):
    rng = np.random.RandomState(42 + node_id * 997)
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


def main():
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    parser = argparse.ArgumentParser()
    parser.add_argument("--round-epochs", type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=0.01)
    parser.add_argument("--batch",        type=int,   default=32)
    parser.add_argument("--global-model", type=str,   default="/root/global_model.npy")
    parser.add_argument("--arch",         type=str,   default=None,
                        help="Comma-separated layer sizes, e.g. --arch 3,16,8,1. "
                             "Overrides the node_id-based default.")
    args = parser.parse_args()

    node_id = get_node_id()
    label   = f"FPGA-NODE{node_id}"

    print(f"[{label}] FL_CLIENT_START node_id={node_id}", flush=True)

    # ── Build model for this node's architecture ──────────────────────────────
    if args.arch is not None:
        arch = [int(x) for x in args.arch.split(",")]
    else:
        arch = NODE_ARCHS.get(node_id, DEFAULT_ARCH)
    model      = FlexDNN(arch)
    arch_b64   = model.arch_descriptor          # e.g. "3,16,1"
    n_params   = model.NumParameters()

    print(f"[{label}] arch={arch_b64}  params={n_params}", flush=True)

    if os.path.exists(args.global_model):
        flat = np.load(args.global_model).flatten()
        if flat.size == n_params:
            model.para = flat.reshape(-1, 1).astype(np.float64)
            model._unpack_para()
            print(f"[{label}] loaded global model shape={flat.shape}", flush=True)
        else:
            print(f"[{label}] global model size mismatch "
                  f"({flat.size} vs {n_params}) — random init", flush=True)
    else:
        print(f"[{label}] no global model — random init", flush=True)

    X, y    = make_dataset(node_id)
    updater = SGD(model.NumParameters(), X, y)

    # ── Task embedding (computed once from local data statistics) ─────────────
    task_vec  = make_task_embedding(X, y, node_id)   # float32[16]
    task_b64  = base64.b64encode(task_vec.tobytes()).decode("ascii")

    print(f"[{label}] dataset: {len(X)} samples | "
          f"round_epochs={args.round_epochs} lr={args.lr}", flush=True)

    round_num = 0
    while _running:
        round_num += 1
        print(f"[{label}] FL_ROUND_START round={round_num}", flush=True)

        loss = train_round(model, updater, X, y,
                           args.round_epochs, args.batch, args.lr)

        np.save(WEIGHTS_PATH, model.para)

        flat = model.para.flatten().astype(np.float32)
        w_b64 = base64.b64encode(flat.tobytes()).decode("ascii")

        # These three lines are parsed by the host fedml_manager
        print(f"FL_ARCH:{arch_b64}",  flush=True)
        print(f"FL_TASK:{task_b64}",  flush=True)
        print(f"FL_WEIGHTS_B64:{w_b64}", flush=True)
        print(f"[{label}] FL_ROUND_DONE round={round_num} loss={loss:.6f}",
              flush=True)

    print(f"[{label}] FL_CLIENT_EXIT rounds_completed={round_num}", flush=True)


if __name__ == "__main__":
    main()
