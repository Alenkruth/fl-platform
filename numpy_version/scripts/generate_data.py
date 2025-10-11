"""Generate synthetic dataset and save as CSV.

Produces a CSV with 3 features (two random features + bias) and a continuous target y.
"""
import numpy as np
import pandas as pd
import argparse

def generate(n=500, seed=0, out='data.csv'):
    np.random.seed(seed)
    x1 = np.random.randn(n, 1)
    x2 = np.random.randn(n, 1)
    bias = np.ones((n,1))
    X = np.hstack([x1, x2, bias])
    # underlying true coefficients
    w = np.array([[2.0], [3.0], [4.0]])
    y = X @ w + np.random.randn(n,1) * 0.1
    df = pd.DataFrame(np.hstack([X, y]), columns=['x1','x2','bias','y'])
    df.to_csv(out, index=False)
    print(f'Wrote {out} with {n} rows')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='data.csv')
    args = parser.parse_args()
    generate(n=args.n, seed=args.seed, out=args.out)
