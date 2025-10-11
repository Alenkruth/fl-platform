import os
import sys
import math
import numpy as np

    # ensure repository root is on sys.path so packages `pytorch_version` and `numpy_version` are importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Register minimal flags expected by original FLplatform code (avoid import-time errors)
import gflags
try:
    gflags.DEFINE_integer('l1_lambda', 0, 'regularization parameter for l1')
except Exception:
    # flag may already be defined; ignore
    pass

from numpy_version.Model.LinearModel import LinearModel as NpLinear
from numpy_version.Model.logsticModel import LogisticModel as NpLogistic
from numpy_version.Model.DNNModel import DNNModel as NpDNN

from pytorch_version.Model.LinearModel import LinearModel as RefLinear
from pytorch_version.Model.logsticModel import LogisticModel as RefLogistic

import torch
import torch.nn.functional as F
from pytorch_version.Model.DNNModel import DNNModel as TorchDNN
# Ensure FLAGS expected by original FLplatform modules exist with safe defaults


def generate_data(n=100, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, 2)
    x = np.concatenate((x, np.ones((n,1))), axis=1)
    y_reg = x @ np.array([[2.0],[3.0],[4.0]]) + np.random.randn(n,1) * 0.1
    y_bin = (y_reg > np.median(y_reg)).astype(float)
    return x, y_reg, y_bin


def flatten_params_from_torch(tmodel):
    parts = []
    for name, p in tmodel.named_parameters():
        arr = p.detach().cpu().numpy()
        # transpose weight matrices to match numpy model packing where weights are (in, out)
        if arr.ndim == 2:
            arr = arr.T
        parts.append(arr.flatten())
    return np.concatenate(parts).reshape(-1,1)


def set_np_dnn_para_from_torch(npmodel, tmodel):
    # Unpack torch params and assign to numpy model arrays
    # Order: linear1.weight, linear1.bias, linear2.weight, linear2.bias
    params = [p.detach().cpu().numpy() for _, p in tmodel.named_parameters()]
    w1 = params[0].T
    b1 = params[1].reshape(1, -1)
    w2 = params[2].T
    b2 = params[3].reshape(1, -1)
    npmodel.W1 = w1.copy()
    npmodel.b1 = b1.copy()
    npmodel.W2 = w2.copy()
    npmodel.b2 = b2.copy()
    npmodel._pack_para()


def test_linear_and_logistic_match_numpy_against_ref():
    seeds = [0, 1]
    for seed in seeds:
        X, y_reg, y_bin = generate_data(n=100, seed=seed)
        # Linear
        a = np.random.RandomState(seed).randn(RefLinear(3).NumParameters(), 1)
        ref = RefLinear(3)
        npmod = NpLinear(3)
        ref.para = a.copy()
        npmod.para = a.copy()
        lr = 0.01
        # one gradient step
        g_ref = ref.PrecomputeCoefficients(ref.para, X, y_reg)
        g_np = npmod.PrecomputeCoefficients(npmod.para, X, y_reg)
        # update
        ref.para = ref.para - lr * g_ref
        npmod.para = npmod.para - lr * g_np
        lref = ref.ComputeLoss(ref.para, X, y_reg)
        lnp = npmod.ComputeLoss(npmod.para, X, y_reg)
        assert np.isclose(lref, lnp, rtol=1e-6, atol=1e-8)

        # Logistic
        a = np.random.RandomState(seed+10).randn(RefLogistic(3).NumParameters(), 1)
        ref2 = RefLogistic(3)
        npmod2 = NpLogistic(3)
        ref2.para = a.copy()
        npmod2.para = a.copy()
        g_ref2 = ref2.PrecomputeCoefficients(ref2.para, X, y_bin)
        g_np2 = npmod2.PrecomputeCoefficients(npmod2.para, X, y_bin)
        ref2.para = ref2.para - lr * g_ref2
        npmod2.para = npmod2.para - lr * g_np2
        lref2 = ref2.ComputeLoss(ref2.para, X, y_bin)
        lnp2 = npmod2.ComputeLoss(npmod2.para, X, y_bin)
        assert np.isclose(lref2, lnp2, rtol=1e-6, atol=1e-8)


def test_dnn_match_torch(seed=0):
    # small DNN classification test (out_size=2)
    X, y_reg, y_bin = generate_data(n=50, seed=seed)
    # convert labels to 0/1 integers
    y_cls = y_bin.flatten().astype(int)

    # Torch model
    torch.manual_seed(seed)
    tmodel = TorchDNN(in_size=3, hidden_size=8, out_size=2)

    # Numpy model
    npmodel = NpDNN(in_size=3, hidden_size=8, out_size=2)

    # copy parameters from torch to numpy to ensure identical initialization
    set_np_dnn_para_from_torch(npmodel, tmodel)

    # prepare torch tensors
    tx = torch.tensor(X, dtype=torch.float32)
    ty = torch.tensor(y_cls, dtype=torch.long)

    # compute torch loss and gradients
    t_out = tmodel(tx)
    t_loss = F.cross_entropy(t_out, ty)
    t_loss.backward()

    # collect torch gradients in same flattened order
    grad_parts = []
    for name, p in tmodel.named_parameters():
        g = p.grad.detach().cpu().numpy()
        if g.ndim == 2:
            g = g.T
        grad_parts.append(g.flatten())
    tgrad = np.concatenate(grad_parts).reshape(-1,1)

    # numpy grad via PrecomputeCoefficients
    npgrad = npmodel.PrecomputeCoefficients(npmodel.para, X, y_cls.reshape(-1,1))

    # both grads should be numerically close
    assert np.allclose(tgrad, npgrad, atol=1e-6, rtol=1e-6)

    # perform identical manual SGD update with same lr
    lr = 0.01
    # update torch params manually
    idx = 0
    for name, p in tmodel.named_parameters():
        arr = p.data.detach().cpu().numpy()
        sz = arr.size
        # tgrad stores gradients in numpy-model orientation (weights are (in, out)).
        g_flat = tgrad[idx:idx+sz].flatten()
        if arr.ndim == 2:
            # reshape to numpy orientation (in, out) then transpose to torch orientation (out, in)
            g_np = g_flat.reshape((arr.shape[1], arr.shape[0]))
            g_torch = g_np.T
        else:
            g_torch = g_flat.reshape(arr.shape)
        p.data = torch.from_numpy((arr - lr * g_torch)).type_as(p.data)
        idx += sz

    # update numpy model
    npmodel.para = npmodel.para - lr * npgrad

    # compute final losses and compare
    with torch.no_grad():
        t_out2 = tmodel(tx)
        t_loss2 = F.cross_entropy(t_out2, ty).item()

    np_loss2 = npmodel.ComputeLoss(npmodel.para, X, y_cls.reshape(-1,1))

    assert np.isclose(t_loss2, np_loss2, rtol=1e-6, atol=1e-8)
