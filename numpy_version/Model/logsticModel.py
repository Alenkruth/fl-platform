"""NumPy LogisticModel

Assumptions:
- Binary classification using sigmoid activation and binary cross-entropy loss.
- `.para` is a column vector; `PrecomputeCoefficients` returns gradient with same shape.

Notes:
- Accuracy computed by thresholding sigmoid at 0.5 in the local runner.
"""

import numpy as np


class LogisticModel:
    def __init__(self, n_coords):
        self.n_coords = n_coords
        self.para = np.zeros((n_coords, 1), dtype=float)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def ComputeLoss(self, para, x, y):
        z = x @ para
        # binary cross-entropy
        eps = 1e-9
        p = self.sigmoid(z)
        return np.mean(- y * np.log(p + eps) - (1 - y) * np.log(1 - p + eps))

    def ProximalOperator(self, para, gamma):
        def operator(n):
            sign = 1 if n > 0 else -1
            return  sign * max(abs(n) - gamma, 0)
        flat = np.array([operator(v) for v in para.flatten()])
        return flat.reshape(para.shape)

    def NumParameters(self):
        return self.n_coords

    def PrecomputeCoefficients(self, para, x, y):
        p = self.sigmoid(x @ para)
        return x.T @ (p - y)
