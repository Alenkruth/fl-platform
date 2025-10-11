"""MPI Worker trainer

Assumptions:
- Worker computes local gradient and exchanges with server using blocking calls.
"""

from mpi4py import MPI
import numpy as np


class WorkerTrainer:
    """Worker that computes local gradients and sends them to the server.

    Protocol (synchronous):
    - Worker computes gradient over its local batch and sends to server.
    - Waits to receive updated parameters from server and replaces local parameters.
    """
    def __init__(self, model, X, y, server_rank=0):
        self.model = model
        self.X = X
        self.y = y
        self.comm = MPI.COMM_WORLD
        self.server = server_rank

    def run_epoch(self, epoch=0):
        # compute full-batch gradient here for simplicity
        grad = self.model.PrecomputeCoefficients(self.model.para, self.X, self.y)
        # send to server
        self.comm.Send([grad, MPI.DOUBLE], dest=self.server, tag=100+epoch)

        # receive updated parameters
        recv_buf = np.empty_like(self.model.para)
        self.comm.Recv([recv_buf, MPI.DOUBLE], source=self.server, tag=200+epoch)
        self.model.para = recv_buf
