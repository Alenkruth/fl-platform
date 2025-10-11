#!/usr/bin/env bash
set -euo pipefail

# Run all permutations of models and optimizers for local_federated.py
# Each run uses --num-workers 1 as requested. Adjust --epochs/--n/--lr below if you want longer runs.

# Location of this script: FLplatform_npy/scripts/run_all_local.sh
# It will invoke the local_federated.py located in the parent directory (FLplatform_npy).

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=python3

# configuration - change these if you want
EPOCHS=1
N=200
LR=0.01

for MODEL in least logistic dnn; do
  for OPT in sgd svrg; do
    echo "============================================================"
    echo "Running: model=${MODEL}, optimizer=${OPT}, workers=1, epochs=${EPOCHS}, n=${N}"
    echo "(Script will exit on first error because of set -e)"
    "$PY" "$ROOT_DIR/local_federated.py" --model "$MODEL" --optimizer "$OPT" --num-workers 1 --epochs "$EPOCHS" --lr "$LR" --n "$N"
    echo
  done
done

echo "All runs completed."
