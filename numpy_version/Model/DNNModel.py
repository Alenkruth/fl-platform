"""NumPy DNNModel (2-layer)

This is a minimal NumPy implementation of a 2-layer neural network (in -> hidden -> out) using ReLU.

Assumptions:
- Uses MSE loss for simplicity (suitable for the synthetic regression data included).
- Parameters are flattened into `para` for compatibility with the updater interface.

Potential pitfalls:
- Not optimized for performance. For classification use-cases replace MSE with cross-entropy.
Deviations from PyTorch:
- Weight initialization attempts to match PyTorch's kaiming_uniform behavior, but exact per-weight values can differ unless the RNG
    and seeding are synchronized.
- Backprop/gradient expressions are implemented explicitly; numerical differences on the order of 1e-6 are expected versus PyTorch.
"""

import numpy as np


class DNNModel:
    """A minimal 2-layer (in->hidden->out) neural network implemented in NumPy.
    Uses ReLU activation and MSE loss for regression (keeps API similar to original).
    """
    def __init__(self, in_size, hidden_size, out_size):
        # weights - initialize to match PyTorch's default patterns for Linear layers
        # PyTorch nn.Linear uses kaiming_uniform_ for weights (with nonlinearity=relu) and
        # uniform_(-1/sqrt(fan_in), 1/sqrt(fan_in)) for biases.
        def kaiming_uniform(shape):
            fan = shape[0]
            bound = np.sqrt(6.0 / fan)  # matches PyTorch kaiming_uniform for relu
            return np.random.uniform(-bound, bound, size=shape)

        def bias_uniform(fan):
            bound = 1.0 / np.sqrt(fan)
            return np.random.uniform(-bound, bound, size=(1,))

        self.W1 = kaiming_uniform((in_size, hidden_size))
        self.b1 = np.random.uniform(-1.0/np.sqrt(in_size), 1.0/np.sqrt(in_size), size=(1, hidden_size))
        self.W2 = kaiming_uniform((hidden_size, out_size))
        self.b2 = np.random.uniform(-1.0/np.sqrt(hidden_size), 1.0/np.sqrt(hidden_size), size=(1, out_size))
        # pack parameters into a para vector for compatibility with updater expected interface
        self._pack_para()

    def _pack_para(self):
        # flatten all params into a single column vector
        parts = [self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten()]
        self.para = np.concatenate([p for p in parts]).reshape(-1, 1)

    def _unpack_para(self):
        # unpack self.para back into W1,b1,W2,b2
        idx = 0
        sW1 = self.W1.size
        self.W1 = self.para[idx:idx+sW1].reshape(self.W1.shape)
        idx += sW1
        sb1 = self.b1.size
        self.b1 = self.para[idx:idx+sb1].reshape(self.b1.shape)
        idx += sb1
        sW2 = self.W2.size
        self.W2 = self.para[idx:idx+sW2].reshape(self.W2.shape)
        idx += sW2
        sb2 = self.b2.size
        self.b2 = self.para[idx:idx+sb2].reshape(self.b2.shape)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        out = a1 @ self.W2 + self.b2
        return z1, a1, out

    def softmax(self, x):
        # stable softmax along last axis
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def ComputeLoss(self, para, X, y):
        # If `out` is multi-dimensional (C>1) we compute softmax + cross-entropy.
        # If out_size == 1 we fall back to MSE (regression).
        self.para = para
        self._unpack_para()
        _, _, out = self.forward(X)
        if out.shape[1] == 1:
            err = out - y
            return float((err**2).mean())
        # y may be provided as column vector of integers or one-hot; normalize to ints
        if y.ndim > 1 and y.shape[1] == 1:
            y_idx = y.flatten().astype(int)
        else:
            y_idx = y.flatten().astype(int)
        probs = self.softmax(out)
        eps = 1e-9
        nll = -np.log(probs[np.arange(len(y_idx)), y_idx] + eps)
        return float(np.mean(nll))

    def ProximalOperator(self, para, gamma):
        # no proximal op for DNN in this simplified port
        return para

    def NumParameters(self):
        return self.para.shape[0]

    def PrecomputeCoefficients(self, para, X, y):
        # return flattened gradient: dLoss/dPara
        self.para = para
        self._unpack_para()
        z1, a1, out = self.forward(X)
        N = X.shape[0]

        if out.shape[1] == 1:
            # MSE regression gradient (kept for compatibility)
            d_out = (2.0 / N) * (out - y)
            dW2 = a1.T @ d_out
            db2 = np.sum(d_out, axis=0, keepdims=True)
            da1 = d_out @ self.W2.T
            dz1 = da1 * self.relu_grad(z1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)
        else:
            # Cross-entropy with softmax gradient
            # y may be ints in column vector form
            if y.ndim > 1 and y.shape[1] == 1:
                y_idx = y.flatten().astype(int)
            else:
                y_idx = y.flatten().astype(int)
            probs = self.softmax(out)
            # gradient of NLL w.r.t. logits: (probs - y_onehot) / N
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(N), y_idx] = 1.0
            d_logits = (probs - y_onehot) / N

            dW2 = a1.T @ d_logits
            db2 = np.sum(d_logits, axis=0, keepdims=True)

            da1 = d_logits @ self.W2.T
            dz1 = da1 * self.relu_grad(z1)

            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

        grads = [dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()]
        gvec = np.concatenate([g for g in grads]).reshape(-1, 1)
        return gvec
