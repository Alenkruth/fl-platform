numpy_version — Design notes, API contract, and pitfalls
=========================================================

This document collects design decisions, API contracts, rationales, and potential pitfalls for
the NumPy port in `numpy_version`.

Overview
--------
numpy_version is a NumPy-based reimplementation of selected pieces from the original
`pytorch_version` project. It aims to preserve the high-level training and federated
protocols while removing the PyTorch dependency for environments where using PyTorch is
undesirable or unavailable.

Primary goals
- Be minimal and easy to read: small, explicit NumPy implementations of models and updaters.
- Preserve API contracts so tests and trainers can interchangeably use models from either
  the original PyTorch tree (`pytorch_version`) or the NumPy port (`numpy_version`).
- Provide two federation backends:
  - local: multiprocessing-based server + workers (easy to run locally without mpirun)
  - mpi: mpirun / mpi4py-based server + workers (keeps original MPI semantics)

Key files and responsibilities
- `Model/` — NumPy models. Each model implements a small contract (see below).
- `Updater/` — Updaters/optimizers. Provide simple interfaces used by workers and the server.
- `Trainer/` — MPI trainer classes (ServerTrainer, WorkerTrainer) used by `mpi_federated.py`.
- `local_federated.py` — multiprocessing runner used for local experiments.
- `mpi_federated.py` — MPI runner; server rank 0, workers 1..N.
- `cli.py` — centralized CLI parser used by all runners so flags remain consistent.
- `run_federated.py` — dispatcher that chooses between `local` and `mpi` modes with `--mode`.

Model API contract
------------------
All models in `Model/` expose the following methods / properties:

- `.para` (numpy column vector) — flattened parameters used by trainers and updaters.
- `ComputeLoss(para, X, y)` — compute scalar loss for diagnostics.
- `PrecomputeCoefficients(para, X, y)` — return flattened gradient vector compatible with `.para`.
- `NumParameters()` — number of parameters.

Rationale: keeping a small, consistent contract lets trainers and updaters operate on flattened
vectors, which simplifies communication (MPI/Pipe) and reduces the need for complex
serialization.

Updater/Optimizer design
------------------------
- Worker-side updaters (SGD, SVRG) compute local gradients by calling
  `model.PrecomputeCoefficients` and send them to the server.
- Server-side optimization is stateful: a `SGDOptimizer` (in `Updater/optim.py`) holds
  momentum buffers and applies weight decay/dampening/nesterov semantics. This keeps
  worker code simple (stateless) and keeps optimizer state centralized on the server.

Rationale: centralizing optimizer state is a common federated pattern (synchronous
parameter server). It avoids synchronizing optimizer buffers across clients.

CLI and running modes
---------------------
- Use `run_federated.py --mode local` for local multiprocessing runs.
- Use `run_federated.py --mode mpi` under `mpirun` for MPI runs.
- The centralized `cli.py` exposes the same flags for both modes: model, optimizer, lr,
  momentum, weight_decay, dampening, nesterov, num-workers, epochs, and n (dataset size).

Design choices and rationale
----------------------------
- Flat parameter vectors: simplifies comms (no pickling of complex objects), easy
  to validate against PyTorch flattened parameters in tests.
- Server-side optimizer state: mirrors real-world federated settings where the
  server updates central parameters and holds optimizer history.
- Two runners (MPI and local): MPI kept to stay close to the original; local
  runner provided to avoid noisy mpirun launcher messages and to run easily
  in constrained environments (CI, notebooks).

Pitfalls and known deviations from PyTorch
----------------------------------------
- Numerical differences: expect small numeric differences due to implementation details,
  initialization, and floating point nuances.
- Determinism: to achieve strict parity you must synchronize RNG seeds, exact
  parameter packing conventions, optimizer internals and minibatch ordering.
- gflags usage: the original code uses `gflags`. Tests register only the minimal
  flags required by the original modules. Access to FLAGS is performed safely in the
  NumPy port to avoid runtime warnings.

Extending and adding features
-----------------------------
- To add other optimizers (Adam, RMSProp), implement server-side optimizer classes
  following the `SGDOptimizer` pattern and ensure their state is stored on the server.
- To add client-side stateful optimizers, update workers to keep optimizer buffers and
  modify the communication protocol to exchange necessary state or apply local updates
  before sending parameter deltas.
- For larger experiments, replace blocking `Send`/`Recv` with `Allreduce` for efficient
  gradient aggregation.

Testing guidance
-----------------
- Unit tests compare NumPy behaviors with PyTorch as a reference for a small DNN and
  linear/logistic models. They check gradient and loss closeness with small tolerances.
- If you add features, update tests to cover the new optimizer behavior and multi-epoch
  dynamics.

Contact and maintenance notes
-----------------------------
This port is intended as a pragmatic, readable alternative to the PyTorch code where
the core ideas of federated learning are preserved while depending only on NumPy.
If you find mismatches or want stronger parity, we can add synchronized RNG, bitwise
parameter copies from Torch to NumPy, and extended optimizer tests.
