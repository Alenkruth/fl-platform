FLplatform_npy
=================

This folder contains a NumPy/SciPy port of the original `FLplatform` project designed to remove the PyTorch dependency while keeping the same high-level APIs for models and updaters. It also contains an MPI-based federated runner so you can spawn multiple processes and simulate federated training with an explicit server and multiple workers.

Files added
- `main.py` - single-process demo runner (no MPI). Good for quick experiments and debugging.
- `mpi_federated.py` - MPI orchestration entrypoint. Use `mpirun` to spawn server + worker processes.
- `Model/` - NumPy implementations of `LinearModel`, `logisticModel`, and a minimal `DNNModel` (2-layer ReLU net).
- `Updater/` - `SGD` and `SVRG` updaters that call model gradient functions.
- `Trainer/ServerTrainer.py` - simple synchronous parameter server: receives gradients from workers, averages, updates global parameters, and sends new parameters back.
- `Trainer/WorkerTrainer.py` - worker logic: computes local gradient and exchanges parameters with server.
 - `scripts/generate_data.py` - generate synthetic CSV dataset for experiments.
 - `scripts/train_and_predict.py` - train a `LinearModel` on CSV data and save predictions.

Rationale and design choices
- Maintain API compatibility: models expose `.para` (flattened parameter vector), `ComputeLoss`, `PrecomputeCoefficients`, and `ProximalOperator` to keep updaters and trainers simple and similar to the originals.
- Simpler DNN: Replaced PyTorch Module with a deterministic NumPy 2-layer network. This keeps the intent (hidden layer + activation + output) while removing heavy framework dependency.
- MPI retained: You asked to keep MPI support. The `mpi_federated.py` script organizes ranks as server (rank 0) and workers (ranks 1..N). Communication is done with `Send`/`Recv` using `MPI.DOUBLE` and numpy buffers.
- Synchronous protocol: for clarity and determinism, the server waits for gradient vectors from all workers every epoch, averages them, performs one SGD step, and returns updated parameters. This is easy to understand, debug, and reproduce.

How to run
1) Single-process demo (no MPI):
```bash
python3 numpy_version/main.py --model least --epochs 20 --batch 32 --lr 0.01
```

Data generation and prediction
--------------------------------

1) Generate synthetic data (CSV):
```bash
python3 numpy-version/scripts/generate_data.py --n 500 --seed 0 --out data.csv
```

2) Train and save predictions:
```bash
python3 numpy-version/scripts/train_and_predict.py --data data.csv --epochs 200 --lr 0.01 --out predictions.csv
```

2) MPI federated run (server + 1 worker):
```bash
mpirun -n 2 python3 numpy-version/mpi_federated.py --num-workers 1 --model least --epochs 10 --lr 0.01
```

3) MPI federated run (server + N workers):
```bash
mpirun -n 4 python3 numpy-version/mpi_federated.py --num-workers 3 --model least --epochs 10 --lr 0.01
```

Notes
- Ensure `mpi4py` and `numpy` are installed in your environment. Install with:
```bash
pip install numpy mpi4py
```
- The current implementation uses blocking `Send`/`Recv` for simplicity. This is fine for small experiments. For production/faster runs, consider non-blocking communication or allreduce for gradient aggregation.
- The DNN is a minimal educational implementation and supports MSE loss; extend to cross-entropy and other features if you need classification with deeper nets.

Next steps (optional)
- Add unit tests for each model and updater.
- Replace blocking MPI calls with `Allreduce` for scalable synchronous aggregation.
- Add trainer variants for asynchronous updates or client sampling.

If you want, I can implement any of the next steps or tweak the communication pattern. 

Assumptions, pitfalls, and verification
--------------------------------------

Assumptions I made while porting to NumPy and adding the local federated runner:

- Models: The NumPy `LinearModel`, `LogisticModel`, and `DNNModel` implement the same high-level
	interfaces as in the original `FLplatform` (methods: `.para`, `ComputeLoss`, `PrecomputeCoefficients`, `ProximalOperator`).
	I assumed those methods are what the rest of the code expects.
- Losses: For the DNN I implemented MSE (regression). The original DNN in the PyTorch code used a classification loss
	in training code paths; I matched the original DNN structure (two layers + ReLU) but for regression experiments
	the MSE is simpler and deterministic in NumPy.
- Updaters: `SGD` and `SVRG` call `model.PrecomputeCoefficients(para, X, y)` and expect a flattened gradient vector compatible
	with `model.para`. I implemented the NumPy updaters to follow that contract.
- Federated protocol: The original project used MPI and a larger trainer stack (ServerTrainer/WorkerTrainer with asynchronous
	or more complex flows). For clarity and reproducibility I implemented a synchronous aggregation protocol both for the MPI trainer
	and the local (multiprocessing) runner. This changes execution ordering but preserves the core functionality (workers compute local
	gradients, server aggregates and updates global parameters).

Potential pitfalls and deviations you should note
------------------------------------------------

- Numeric differences: NumPy vs PyTorch (float precision, initialization) will produce different trajectories. Expect small
	deviations in loss/accuracy; this is normal. On simple synthetic data the results should be similar in magnitude.
- Optimization details: The original PyTorch code may have used different defaults (weight initialization, optimizer specifics,
	learning-rate schedules, minibatch ordering). Those were simplified here (deterministic small random init, simple step updates).
- DNN behavior: The NumPy DNN is intentionally small and unoptimized; it lacks advanced stable numerical routines (batchnorm,
	advanced initializers etc.). For classification tasks you may need to change the DNN loss to cross-entropy.
- Federated differences: The synchronous server averages gradients; if the original code used asynchronous updates or momentum, results
	will differ. If you require bit-for-bit equivalence, we'd have to replicate the exact PyTorch training loop (including RNG seeding,
	minibatch order, and optimizer internals) which is beyond a straightforward port.

How to verify the port retains functionality
--------------------------------------------

1) Run the same training permutation on both trees (the original `FLplatform` and `FLplatform_npy`) using the same dataset and
	 comparable hyperparameters (learning rate, epochs, batch size). For local_federated we provide a `--n` sample count to control dataset size.
2) Compare per-worker final loss/accuracy. On synthetic regression data (the generator included) the numbers should be in the same ballpark
	 (e.g., mean squared error within a small factor). For classification (logistic) compare accuracy — expect similar trends though exact
	 percentages may differ.
3) If you need a quantitative tolerance, start with relative difference threshold (e.g., |loss_numpy - loss_torch| / max(loss_torch,1e-6) < 0.2)
	 and relax if necessary.

If you find a specific permutation where results diverge drastically, provide the logs and I will reproduce and adjust the implementations
to reduce the gap (e.g., matching initialization, minibatch RNG, or optimizer step precisely).

Deviations summary (explicit)
------------------------------
- Initialization: RNG seeds and initializers differ slightly unless explicitly synchronized across implementations.
- Loss & scaling: Some gradients are computed per-sample vs per-batch depending on the original implementation; we preserved the original
	`PrecomputeCoefficients` semantics where possible.
- Optimizer internals: PyTorch optimizers carry state and internal scaling; this port implements simpler updates.

Full design notes and rationale are available in `numpy_version/DOCUMENTATION.md` which explains
the model/updater contracts, server-side optimizer decisions, and extension notes.

Detailed documentation
----------------------
For deeper technical notes, API contracts, and design rationale see:

`numpy_version/DOCUMENTATION.md`


Recent changes (optimizer hyperparameters)
-----------------------------------------
- Added server-side optimizer hyperparameter support for both local and MPI federated runners.
	- `local_federated.py` now accepts `--momentum`, `--weight-decay`, `--dampening`, and `--nesterov` and passes them to the server-side `SGDOptimizer` so momentum buffers persist across epochs.
	- `mpi_federated.py` now accepts the same flags and forwards them to `ServerTrainer` which constructs a stateful `SGDOptimizer` when using SGD.

Usage examples (server-side momentum / weight decay):
```bash
# Local multiprocessing runner
python3 numpy-version/local_federated.py --model least --optimizer sgd --num-workers 2 --epochs 10 --lr 0.01 --momentum 0.9 --weight-decay 1e-4

# MPI runner (server + 2 workers)
mpirun -n 3 python3 numpy-version/mpi_federated.py --num-workers 2 --model least --epochs 10 --lr 0.01 --momentum 0.9 --weight-decay 1e-4
```

CLI parsing consistency
----------------------
The runner CLIs accept both dash and underscore variants for common optimizer flags for convenience. You can use either form:

```bash
# either
python3 numpy-version/local_federated.py --momentum 0.9 --weight-decay 1e-4
# or
python3 numpy-version/local_federated.py --momentum 0.9 --weight_decay 1e-4
```

Final notes
-----------
- Server-side optimizer buffers (momentum) are stored only on the server; workers remain stateless and only compute/send gradients.
- Tests were run after these edits and passed (2 tests). Existing gflags warnings remain and are unrelated to these changes.
