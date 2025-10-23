"""NumPy/SciPy port of FLplatform/main.py that runs without PyTorch or MPI.
This is a simplified single-process runner for least squares, logistic, and a small DNN implemented with NumPy.
"""
import numpy as np
import argparse
import time
from Model.LinearModel import LinearModel
from Model.logsticModel import LogisticModel
from Model.DNNModel import DNNModel
from Updater.SGD import SGD
from Updater.SVRG import SVRG


def generate_data_set(seed=0):
    np.random.seed(seed)
    x = np.random.rand(200, 2)
    x = np.concatenate((x, np.ones((200,1), dtype=float)), axis=1)
    y = 2 * x[:, 0:1]+ 3 * x[:, 1:2] + 4 * x[:, 2:3] + np.random.randn(200, 1) * 0.1
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['least','logistic','dnn'], default='least')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    X, y = generate_data_set(seed=1)

    if args.model == 'least':
        X = np.c_[X, np.ones((X.shape[0], 1))] if X.shape[1] == 2 else X
        model = LinearModel(X.shape[1])
        updater = SGD(model.NumParameters(), X, y)
    elif args.model == 'logistic':
        # create binary labels for logistic demo
        y_bin = (y > np.median(y)).astype(float)
        X = np.c_[X, np.ones((X.shape[0], 1))]
        model = LogisticModel(X.shape[1])
        updater = SGD(model.NumParameters(), X, y_bin)
        y = y_bin
    else:
        model = DNNModel(in_size=3, hidden_size=16, out_size=1)
        # make y continuous for regression DNN
        updater = SGD(model.NumParameters(), X, y)

    for epoch in range(args.epochs):
        losses = []
        # simple full-batch gradient step for clarity (mini-batch supported below)
        for i in range(0, X.shape[0], args.batch):
            xb = X[i:i+args.batch]
            yb = y[i:i+args.batch]
            grad = updater.Update(model, xb, yb)
            # apply gradient step
            model.para = model.para - args.lr * grad
            model.para = model.ProximalOperator(model.para, args.lr * 0.0)
            # compute loss
            losses.append(model.ComputeLoss(model.para, X, y))
        print(f'Epoch {epoch+1}/{args.epochs} loss={np.mean(losses):.6f}')


if __name__ == '__main__':
    main()
