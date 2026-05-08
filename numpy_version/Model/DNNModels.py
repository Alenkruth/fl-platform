"""Generalized multi-layer DNN for heterogeneous FireSim FL clients.

FlexDNN supports arbitrary depth via a layer_sizes list [in, h1, h2, ..., out].
All hidden layers use ReLU; the output layer is linear.
Parameters are packed into a flat `para` column vector compatible with the
SGD/SVRG updater interface.

Named variants:
    ShallowDNN()  →  FlexDNN([3, 16, 1])      81 params  (Node 0 default)
    DeepDNN()     →  FlexDNN([3, 16, 8, 1])  209 params  (Node 1 default)
    WideDNN()     →  FlexDNN([3, 32, 1])      161 params

Architecture descriptor (arch_descriptor attribute):
    Comma-separated layer sizes, e.g. "3,16,1".
    Printed by firesim_client.py as FL_ARCH:<descriptor> for the host manager.
"""

import numpy as np


class FlexDNN:
    """Variable-depth MLP with a fixed API matching DNNModel."""

    def __init__(self, layer_sizes: list):
        assert len(layer_sizes) >= 2, "need at least [in_size, out_size]"
        self.layer_sizes = [int(s) for s in layer_sizes]
        self.arch_descriptor = ",".join(str(s) for s in self.layer_sizes)
        self.n_layers = len(self.layer_sizes) - 1

        self.Ws = []
        self.bs = []
        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            bound = np.sqrt(6.0 / fan_in)
            W = np.random.uniform(-bound, bound, size=(fan_in, fan_out))
            b = np.random.uniform(-1.0 / np.sqrt(fan_in),
                                   1.0 / np.sqrt(fan_in), size=(1, fan_out))
            self.Ws.append(W)
            self.bs.append(b)
        self._pack_para()

    # ── Parameter serialization ───────────────────────────────────────────────

    def _pack_para(self):
        parts = []
        for W, b in zip(self.Ws, self.bs):
            parts += [W.flatten(), b.flatten()]
        self.para = np.concatenate(parts).reshape(-1, 1)

    def _unpack_para(self):
        idx = 0
        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            nW = fan_in * fan_out
            nb = fan_out
            self.Ws[i] = self.para[idx:idx + nW].reshape(fan_in, fan_out)
            idx += nW
            self.bs[i] = self.para[idx:idx + nb].reshape(1, fan_out)
            idx += nb

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, X):
        """Returns list of (pre_act, post_act) per layer."""
        activations = []
        h = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = h @ W + b
            a = np.maximum(0, z) if i < self.n_layers - 1 else z
            activations.append((z, a))
            h = a
        return activations

    # ── Loss ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x):
        x = x - np.max(x, axis=1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=1, keepdims=True)

    def ComputeLoss(self, para, X, y):
        self.para = para
        self._unpack_para()
        acts = self.forward(X)
        out  = acts[-1][1]
        if out.shape[1] == 1:
            err = out - y
            return float((err ** 2).mean())
        y_idx = y.flatten().astype(int)
        probs = self._softmax(out)
        return float(-np.log(probs[np.arange(len(y_idx)), y_idx] + 1e-9).mean())

    # ── Gradient ─────────────────────────────────────────────────────────────

    def PrecomputeCoefficients(self, para, X, y):
        self.para = para
        self._unpack_para()
        acts = self.forward(X)
        N    = X.shape[0]
        out  = acts[-1][1]

        if out.shape[1] == 1:
            d = (2.0 / N) * (out - y)
        else:
            y_idx = y.flatten().astype(int)
            probs = self._softmax(out)
            d = probs.copy()
            d[np.arange(N), y_idx] -= 1.0
            d /= N

        h_ins = [X] + [a for _, a in acts[:-1]]
        grads = []
        for i in reversed(range(self.n_layers)):
            dW = h_ins[i].T @ d
            db = np.sum(d, axis=0, keepdims=True)
            grads.insert(0, (dW, db))
            if i > 0:
                d = (d @ self.Ws[i].T) * (acts[i - 1][0] > 0)

        parts = []
        for dW, db in grads:
            parts += [dW.flatten(), db.flatten()]
        return np.concatenate(parts).reshape(-1, 1)

    # ── Updater interface ─────────────────────────────────────────────────────

    def ProximalOperator(self, para, gamma):
        return para

    def NumParameters(self):
        return int(self.para.shape[0])

    def __repr__(self):
        return f"FlexDNN({self.layer_sizes}, params={self.NumParameters()})"


# ── Named variants ────────────────────────────────────────────────────────────

def ShallowDNN():
    """3→16→1, 81 params. Default for Node 0. Matches original DNNModel."""
    return FlexDNN([3, 16, 1])


def DeepDNN():
    """3→16→8→1, 209 params. Default for Node 1."""
    return FlexDNN([3, 16, 8, 1])


def WideDNN():
    """3→32→1, 161 params."""
    return FlexDNN([3, 32, 1])


def arch_from_descriptor(descriptor: str) -> "FlexDNN":
    """Build a FlexDNN from a comma-separated descriptor string, e.g. '3,16,1'."""
    sizes = [int(x) for x in descriptor.strip().split(",")]
    return FlexDNN(sizes)


def num_params_for_arch(layer_sizes: list) -> int:
    """Compute parameter count for a given architecture without instantiating a model."""
    return sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
               for i in range(len(layer_sizes) - 1))


def make_task_embedding(X: "np.ndarray", y: "np.ndarray",
                         node_id: int, dim: int = 16) -> "np.ndarray":
    """Compute a fixed-dim task embedding from local data statistics.

    Called by firesim_client.py before printing FL_TASK:<base64>.
    X: (N, F) feature matrix.  y: (N, 1) target vector.
    """
    feat = [
        float(np.mean(X[:, 0])),   float(np.std(X[:, 0])),
        float(np.mean(X[:, 1])),   float(np.std(X[:, 1])),
        float(np.mean(X[:, 2])),   float(np.std(X[:, 2])),
        float(np.mean(y)),         float(np.std(y)),
        float(np.min(y)),          float(np.max(y)),
        float(node_id),
        float(len(X)),
        float(np.corrcoef(X[:, 0], y.flatten())[0, 1]),
        float(np.corrcoef(X[:, 1], y.flatten())[0, 1]),
        float(np.corrcoef(X[:, 2], y.flatten())[0, 1]),
        0.0,
    ]
    return np.array(feat[:dim], dtype=np.float32)
