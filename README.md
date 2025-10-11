FL_platform repository
======================

This repository contains two implementations of small federated learning examples:

- `pytorch_version` - the original PyTorch-based implementation.
- `numpy_version` (formerly `numpy-version` / `FLplatform_npy`) - a NumPy-based port that mirrors the functionality and provides
  local multiprocessing and simplified MPI runners.

Quick start
-----------

1. Install dependencies (recommended in a virtualenv):

```bash
pip install numpy pandas mpi4py pytest torch
```

2. Generate data and run local federated experiments (NumPy port):

```bash
python3 -m numpy_version.scripts.generate_data --n 500 --out numpy_version/data.csv
python3 -m numpy_version.local_federated --model least --optimizer sgd --num-workers 2 --epochs 5 --lr 0.01 --n 500
```

Running tests
-------------
Unit tests that compare selected computations between the PyTorch and NumPy implementations are in `tests/`.

```bash
pytest -q
```

If tests fail, check that `torch` and other dependencies are installed and that you are running tests from the repository root.
# FL_platform
A flexible FL platform with GUI

Library: requirement.txt

> mpirun -n 4 python -m pytorch_version.main
