"""MPI-based federated runner.

Usage (example):
  mpirun -n 3 python3 FLplatform_npy/mpi_federated.py --num-workers 2 --model least --epochs 5 --lr 0.01

Design:
- Rank 0 is the server. Ranks 1..num_workers are workers. The script accepts --num-workers to set
  how many worker ranks it expects. The total MPI processes must be at least num_workers+1.
"""
import numpy as np
from mpi4py import MPI
from cli import parse_common_args

from Model.LinearModel import LinearModel
from Model.logsticModel import LogisticModel
from Model.DNNModel import DNNModel
from Trainer.ServerTrainer import ServerTrainer
from Trainer.WorkerTrainer import WorkerTrainer

def generate_data_set(seed=0, n=200):
    np.random.seed(seed)
    x = np.random.rand(n, 2)
    x = np.concatenate((x, np.ones((n,1), dtype=float)), axis=1)
    y = 2 * x[:, 0:1]+ 3 * x[:, 1:2] + 4 * x[:, 2:3] + np.random.randn(n, 1) * 0.1
    return x, y

def run_with_args(args=None):
    if args is None:
        args = parse_common_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # determine number of workers
    n_workers = 1 if args.one_worker else args.num_workers
    if size < n_workers + 1:
        if rank == 0:
            raise RuntimeError(f'Not enough MPI processes: need server + {n_workers} workers (total {n_workers+1}), got {size}')
        return

    # Create or load data and partition among workers deterministically
    X, y = generate_data_set(seed=1, n=200)

    if args.model == 'least':
        model_template = LinearModel(X.shape[1])
    elif args.model == 'logistic':
        model_template = LogisticModel(X.shape[1])
    else:
        model_template = DNNModel(in_size=X.shape[1], hidden_size=16, out_size=1)

    if rank == 0:
        # server
        server = ServerTrainer(
            model_template,
            num_workers=n_workers,
            lr=args.lr,
            optimizer_name='sgd',
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            dampening=args.dampening,
            nesterov=args.nesterov,
        )
        server.run(n_epochs=args.epochs)
    else:
        # worker ranks 1..n_workers
        worker_id = rank - 1
        if worker_id >= n_workers:
            # idle ranks (not participating)
            return

        # partition data evenly
        N = X.shape[0]
        per = N // n_workers
        start = worker_id * per
        end = (worker_id+1)*per if worker_id < n_workers-1 else N
        X_local = X[start:end]
        y_local = y[start:end]

        # create a fresh model copy for the worker
        worker_model = type(model_template)(*(getattr(model_template, 'W1').shape[0], 16, 1)) if args.model == 'dnn' else type(model_template)(X_local.shape[1])
        worker = WorkerTrainer(worker_model, X_local, y_local, server_rank=0)
        for ep in range(args.epochs):
            worker.run_epoch(epoch=ep)

def main():
    args = parse_common_args()
    run_with_args(args)


if __name__ == '__main__':
    main()
