"""Simplified SVRG updater for NumPy models.

Notes:
- This implementation computes a full-gradient snapshot in `EpochBegin` and uses it in `Update`.
Deviations from PyTorch:
- SVRG is implemented here in a simplified, single-process form for clarity. If using parallel workers over real distributed data,
  the gradient snapshot and communication would need to be handled via MPI/allreduce to match distributed PyTorch setups.
"""

import numpy as np


class SVRG:
    def __init__(self, coord, x, y):
        self.coord = coord
        self.x = x
        self.y = y
        self.copy_para = None
        self.copy_grad = None

    def EpochBegin(self, model):
        # save snapshot of parameters and compute full gradient
        self.copy_para = model.para.copy()
        self.copy_grad = model.PrecomputeCoefficients(self.copy_para, self.x, self.y)

    def Update(self, model, x, y):
        
        # intialize snapshot if missing
        if self.copy_para is None or self.copy_grad is None:
            self.EpochBegin(model)

        grad1 = model.PrecomputeCoefficients(model.para, x, y)
        grad2 = model.PrecomputeCoefficients(self.copy_para, x, y)
        g = (grad1 - grad2) + self.copy_grad
        return g
