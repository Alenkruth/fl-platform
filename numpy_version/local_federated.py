"""Local multiprocessing federated runner.

This script simulates a synchronous federated learning run using Python `multiprocessing`.

Design / assumptions:
- Server runs in the main process and communicates with each worker over a `Pipe`.
- Each worker receives the global `para`, computes a gradient via `PrecomputeCoefficients`, sends
    the gradient to the server, and receives updated parameters.
- The script accepts CLI options to select the model (`least`, `logistic`, `dnn`), optimizer (`sgd`, `svrg`),
    number of workers, epochs, learning rate, and dataset size.

Potential pitfalls:
- Because this is synchronous and uses full-batch local gradients, behavior will differ from an
    asynchronous or minibatch training loop. Expect numerical/trajectory deviations vs the original PyTorch
    implementation. However, the high-level functionality (worker-local gradients, server aggregation) is preserved.

Deviations from PyTorch implementation (what to expect):
- NumPy uses explicit array operations and different default memory layout; tiny numerical differences are normal.
- Initialization: DNN uses a kaiming-like uniform initializer to match PyTorch closely, but exact RNG states may differ.
- Optimization: SGD here is implemented as a simple parameter subtract; PyTorch optimizers include state (momentum, weight decay) and
    may use different internal scaling for gradients. SVRG is simplified relative to advanced PyTorch optimizers.
- DNN loss: NumPy DNN uses MSE for out_size==1 and cross-entropy for multi-class. If the original used a different loss, update accordingly.
"""
import multiprocessing as mp
import numpy as np
import time

from Model.LinearModel import LinearModel
from Model.logsticModel import LogisticModel
from Model.DNNModel import DNNModel
from Updater.SGD import SGD
from Updater.SVRG import SVRG
from Updater.optim import SGDOptimizer
from cli import parse_common_args


def generate_data_set(seed=0, n=200):
    np.random.seed(seed)
    x = np.random.rand(n, 2)
    x = np.concatenate((x, np.ones((n,1), dtype=float)), axis=1)
    y = 2 * x[:, 0:1]+ 3 * x[:, 1:2] + 4 * x[:, 2:3] + np.random.randn(n, 1) * 0.1
    return x, y


def worker_process(conn, model_name, optimizer_name, X, y, epochs, lr, worker_id):
    # create model instance inside the process based on requested model
    if model_name == 'least':
        model = LinearModel(X.shape[1])
    elif model_name == 'logistic':
        model = LogisticModel(X.shape[1])
    else:
        model = DNNModel(in_size=X.shape[1], hidden_size=16, out_size=1)

    # choose optimizer instance
    if optimizer_name == 'sgd':
        updater = SGD(model.NumParameters(), X, y)
    else:
        updater = SVRG(model.NumParameters(), X, y)

    for ep in range(epochs):
        # receive current parameters from server
        para = conn.recv()
        model.para = para

        # compute gradient using updater
        if optimizer_name == 'sgd':
            grad = updater.Update(model, X, y)
        else:
            # For SVRG, let the worker request epoch snapshot from server by computing local snapshot
            # Here we reuse updater as single-process SVRG (it expects EpochBegin to be called)
            updater.EpochBegin(model)
            grad = updater.Update(model, X, y)

        # send gradient
        conn.send(grad)

        # receive updated parameters
        para = conn.recv()
        model.para = para

    # after training, compute loss (and accuracy for logistic)
    loss = model.ComputeLoss(model.para, X, y)
    if model_name == 'logistic':
        # compute accuracy
        probs = LogisticModel.sigmoid(X @ model.para)
        preds = (probs >= 0.5).astype(int)
        acc = (preds == y).mean()
        print(f'Worker {worker_id}: loss={loss:.6f} accuracy={acc:.4f}')
    else:
        print(f'Worker {worker_id}: loss={loss:.6f}')

    conn.close()


def server_main(conns, model, epochs, lr, optimizer_name='sgd', momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False):
    n_workers = len(conns)

    # create a stateful optimizer on the server if requested
    if optimizer_name == 'sgd':
        optim = SGDOptimizer(model.para.shape, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)
    else:
        optim = None

    for ep in range(epochs):
        # send current params to all workers
        for conn in conns:
            conn.send(model.para)

        # collect grads
        grads = []
        for conn in conns:
            g = conn.recv()
            grads.append(g)

        agg = sum(grads) / len(grads)

        # if optimizer is stateful, use it
        if optim is not None:
            model.para = optim.step(agg, model.para)
        else:
            model.para = model.para - lr * agg

        # send updated params back
        for conn in conns:
            conn.send(model.para)

    for conn in conns:
        conn.close()


def run_with_args(args=None):
    if args is None:
        args = parse_common_args()

    X, y = generate_data_set(seed=1, n=args.n)

    # create a template model in server
    if args.model == 'least':
        server_model = LinearModel(X.shape[1])
    elif args.model == 'logistic':
        server_model = LogisticModel(X.shape[1])
    else:
        server_model = DNNModel(in_size=X.shape[1], hidden_size=16, out_size=1)

    # partition data evenly across workers
    N = X.shape[0]
    per = max(1, N // args.num_workers)

    processes = []
    conns = []
    for i in range(args.num_workers):
        start = i * per
        end = min((i+1)*per, N)
        parent_conn, child_conn = mp.Pipe()
        conns.append(parent_conn)
        p = mp.Process(target=worker_process, args=(child_conn, args.model, args.optimizer, X[start:end], y[start:end], args.epochs, args.lr, i))
        p.start()
        processes.append(p)

    # run server loop in main process (pass optimizer hyperparameters so server optimizer retains state across epochs)
    server_main(conns, server_model, args.epochs, args.lr, args.optimizer, args.momentum, args.weight_decay, args.dampening, args.nesterov)

    # wait for workers
    for p in processes:
        p.join()

    print('Server: Training complete. Final loss:', server_model.ComputeLoss(server_model.para, X, y))


def main():
    args = parse_common_args()
    run_with_args(args)


if __name__ == '__main__':
    main()
