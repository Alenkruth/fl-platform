Tests for FL_platform
======================

This directory contains a small set of unit tests used to verify numerical parity
between the original PyTorch models in `FLplatform` and the NumPy ports in
`numpy-version`.

How to run
----------

From the repository root run:

```bash
pytest -q
```

What they check
---------------

- `test_compare.py` compares one-step gradient computations and parameter updates
  for linear, logistic and a small DNN between the two implementations.

Notes
-----

- Tests require `torch` in the environment because the NumPy implementation is
  validated against PyTorch as a reference. Install it with `pip install torch`.
- Tests allow small numerical tolerances; exact bitwise equivalence is not
  expected due to differences in floating point implementations and non-determinism.
