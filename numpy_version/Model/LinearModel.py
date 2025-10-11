"""NumPy LinearModel

Assumptions:
- Exposes `.para` as a flattened column vector of parameters.
- `ComputeLoss(para, X, y)` returns scalar loss (MSE) using the provided `para`.
- `PrecomputeCoefficients(para, X, y)` returns gradient flattened as a column vector compatible with `para`.

Potential pitfalls:
- Differences in initialization or numeric precision vs PyTorch will change training trajectories.
"""

import numpy as np


class LinearModel:
    def __init__(self, n_coords):
        self.n_coords = n_coords
        # initialize parameters small
        self.para = np.zeros((n_coords, 1), dtype=float)

    def ComputeLoss(self, para, x , y):
        y_pre = x @ para - y
        return (y_pre.T @ y_pre)[0][0] / np.shape(x)[0]

    def ProximalOperator(self, para, gamma):
        # soft-thresholding (L1)
        def operator(n):
            sign = 1 if n > 0 else -1
            return  sign * max(abs(n) - gamma, 0)
        flat = np.array([operator(v) for v in para.flatten()])
        return flat.reshape(para.shape)

    def NumParameters(self):
        return self.n_coords

    def PrecomputeCoefficients(self, para, x, y):
        # Return gradient (same as original FLplatform implementation - not averaged)
        return x.T @ (x @ para - y)
