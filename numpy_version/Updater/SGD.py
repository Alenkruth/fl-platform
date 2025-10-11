"""Simple SGD updater for NumPy models.

Assumptions:
- Model exposes `PrecomputeCoefficients(para, X, y)` which returns flattened gradient vector.
Deviations from PyTorch:
- This SGD implementation applies a direct parameter step p = p - lr * grad. PyTorch's SGD may divide by batch size depending on loss
    implementation and could include momentum/weight decay when configured.
"""

import numpy as np


class SGD:
    def __init__(self, coord, x, y):
        self.coord = coord
        self.x = x
        self.y = y

    def EpochBegin(self, model):
        pass

    def Update(self, model, x, y):
        # compute gradient vector using model's PrecomputeCoefficients
        # Ensure gradient is averaged over the minibatch (PyTorch convention).
        g = model.PrecomputeCoefficients(model.para, x, y)
        return g
