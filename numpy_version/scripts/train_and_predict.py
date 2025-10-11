"""Train a LinearModel on a CSV dataset and save predictions.

Expects a CSV with columns: x1,x2,bias,y. Saves `predictions.csv` with columns `y_true,y_pred`.
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys

# ensure FLplatform_npy is on sys.path so `from Model...` imports succeed when running from repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(SCRIPT_DIR)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from Model.LinearModel import LinearModel


def train(X, y, epochs=100, lr=0.01):
    model = LinearModel(X.shape[1])
    for ep in range(epochs):
        grad = model.PrecomputeCoefficients(model.para, X, y)
        model.para = model.para - lr * grad
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data.csv')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--out', default='predictions.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X = df[['x1','x2','bias']].to_numpy()
    y = df[['y']].to_numpy()

    model = train(X, y, epochs=args.epochs, lr=args.lr)

    y_pred = X @ model.para
    out_df = pd.DataFrame(np.hstack([y, y_pred]), columns=['y_true','y_pred'])
    out_df.to_csv(args.out, index=False)
    print(f'Saved predictions to {args.out}')


if __name__ == '__main__':
    main()
