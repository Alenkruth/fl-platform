"""Synchronous parameter server (MPI-based)

Assumptions:
- Server receives gradients from workers, averages them, applies a simple SGD step, and sends updated parameters back.
- Uses blocking `Recv`/`Send` for clarity and determinism.

Potential pitfalls:
- Blocking communication is simple but can be slow; consider `Allreduce` for serverless aggregation.
"""

from mpi4py import MPI
import numpy as np


class ServerTrainer:
    """Simple synchronous parameter server.

    Protocol (synchronous):
    - Server waits to receive gradient vectors from each worker for an iteration.
    - Aggregates gradients (simple average) and updates global parameters.
    - Broadcasts updated parameters back to workers.
    """
    def __init__(self, model, num_workers, lr=1e-3, optimizer_name='sgd', momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False):
        self.model = model
        self.num_workers = num_workers
        self.lr = lr
        self.comm = MPI.COMM_WORLD

        # optional stateful optimizer (server-side)
        self.optimizer_name = optimizer_name
        self.optimizer = None
        if optimizer_name == 'sgd':
            # lazy import to avoid circular imports at module import time
            from Updater.optim import SGDOptimizer
            self.optimizer = SGDOptimizer(self.model.para.shape, lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, nesterov=nesterov)

    def run(self, n_epochs=1):
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()
        for ep in range(n_epochs):
            # receive gradients from all workers
            grads = []
            for w in range(1, self.num_workers+1):
                # probe to get size and then receive
                recv_buf = np.empty_like(self.model.para)
                self.comm.Recv([recv_buf, MPI.DOUBLE], source=w, tag=100+ep)
                grads.append(recv_buf)

            # aggregate
            agg = sum(grads) / len(grads)

            # update parameters (use stateful optimizer if present)
            if self.optimizer is not None:
                self.model.para = self.optimizer.step(agg, self.model.para)
            else:
                self.model.para = self.model.para - self.lr * agg

            # broadcast updated parameters back to workers
            for w in range(1, self.num_workers+1):
                self.comm.Send([self.model.para, MPI.DOUBLE], dest=w, tag=200+ep)
