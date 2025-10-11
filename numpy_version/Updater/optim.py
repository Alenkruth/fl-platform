"""Optimizer implementations for FLplatform_npy that mimic PyTorch optimizer internals.

Currently provides SGD with momentum, weight decay, dampening and Nesterov.

This optimizer is intended to be used on the server side where global parameter state
and optimizer buffers are stored. Workers send gradients; server aggregates and passes
the aggregated gradient to this optimizer for a stateful update.
"""
import numpy as np


class SGDOptimizer:
    def __init__(self, param_shape, lr=1e-3, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov
        # state buffer (velocity) initialized to zeros with same shape as params
        self.buffer = np.zeros(param_shape, dtype=float)

    def step(self, grad, param):
        """Apply one SGD step to param in-place using grad.

        grad and param are expected to be column vectors of the same shape.
        This follows PyTorch's SGD implementation semantics:
        - apply weight decay: grad = grad + weight_decay * param
        - update buffer: buf = momentum * buf + (1 - dampening) * grad (we use the PyTorch simplified version buf = momentum*buf + grad)
        - compute d_p: if nesterov: d_p = grad + momentum * buf else d_p = buf
        - param = param - lr * d_p
        """
        # apply weight decay
        if self.weight_decay != 0.0:
            grad = grad + self.weight_decay * param

        if self.momentum != 0.0:
            # PyTorch uses: buf = momentum * buf + grad
            self.buffer = self.momentum * self.buffer + grad
            if self.nesterov:
                d_p = grad + self.momentum * self.buffer
            else:
                d_p = self.buffer
        else:
            d_p = grad

        # update parameters
        param -= self.lr * d_p

        return param
