"""NumPy/SciPy port of FLplatform/main.py that runs without PyTorch or MPI.
This is a simplified single-process runner for least squares, logistic, and a small DNN implemented with NumPy.
"""
import numpy as np
import argparse
import time
from datetime import datetime
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


def train_test_split(X, y, test_size=0.2, seed=42):
    """Split data into train and test sets"""
    np.random.seed(seed)
    n = X.shape[0]
    indices = np.random.permutation(n)
    split_idx = int(n * (1 - test_size))
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def model_predict(model, X):
    pred = None

    for method in ['Inference', 'predict', 'forward']:
        if hasattr(model, method):
            fn = getattr(model, method)
            try:
                pred = fn(X)
                break
            except Exception:
                continue

    if isinstance(pred, tuple):
        pred = pred[-1] # dnn forward return (z1, a1, out)

    if pred is None and hasattr(model, 'W1'):
        z1 = X @ model.W1 + model.b1
        a1 = np.maximum(0, z1) # ReLU
        pred = a1 @ model.W2 + model.b2

    if pred is None and hasattr(model, 'para'):
        pred = X @ model.para
        if hasattr(model, 'sigmoid'):
            pred = 1 / (1 + np.exp(-np.clip(pred, -500, 500)))

    if isinstance(pred, tuple):
        pred = pred[0]
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)

    return pred.flatten() if pred.ndim > 1 else pred

# def model_predict(model, X):
#     """Get predictions from model - works with different model types"""
#     # Try different methods that models might have
#     if hasattr(model, 'Inference'):
#         return model.Inference(X)
#     elif hasattr(model, 'predict'):
#         return model.predict(X)
#     elif hasattr(model, 'forward'):
#         return model.forward(X)
#     else:
#         # Fallback: compute manually based on model type
#         if hasattr(model, 'W1'):  # DNN model
#             # Forward pass for DNN
#             z1 = X @ model.W1 + model.b1
#             a1 = np.maximum(0, z1)  # ReLU
#             pred = a1 @ model.W2 + model.b2
#             return pred.flatten() if pred.ndim > 1 else pred
#         else:
#             # Linear/Logistic model - assume it's X @ weights
#             # The model.para likely contains the weights
#             pred = X @ model.para
#             if hasattr(model, 'sigmoid'):  # Logistic model
#                 pred = 1 / (1 + np.exp(-np.clip(pred, -500, 500)))
# 
#             # --- ADD BELOW to normalize pred output ---
#             if isinstance(pred, tuple):
#                 # Some models (like DNNModel) might return (output, activations)
#                 pred = pred[0]
#             if not isinstance(pred, np.ndarray):
#                 pred = np.array(pred)
#             return pred.flatten() if pred.ndim > 1 else pred
#             # return pred.flatten() if pred.ndim > 1 else pred


def compute_accuracy(model, X, y):
    """Compute accuracy for classification models"""
    pred = model_predict(model, X)
    pred_class = (pred > 0.5).astype(float)
    return np.mean(pred_class == y.flatten())


def compute_r2_score(model, X, y):
    """Compute R² score for regression models"""
    pred = model_predict(model, X)
    y_flat = y.flatten()
    pred_flat = pred.flatten() if pred.ndim > 1 else pred
    ss_res = np.sum((y_flat - pred_flat) ** 2)
    ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
    return 1 - (ss_res / ss_tot)

def safe_loss(loss_val):
    if isinstance(loss_val, tuple):
        return float(loss_val[0])
    return float(loss_val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['least','logistic','dnn'], default='least')
    parser.add_argument('--optimizer', choices=['sgd','svrg'], default='sgd',
                        help='Optimizer: sgd or svrg')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    # Print configuration header
    print(f"\n{'='*60}")
    print(f"MODEL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model:       {args.model.upper()}")
    print(f"Optimizer:   {args.optimizer.upper()}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch Size:  {args.batch}")
    print(f"Learning Rate: {args.lr}")
    print(f"{'='*60}\n")

    # Generate dataset
    print("Generating synthetic dataset...")
    X, y = generate_data_set(seed=1)
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

    # Prepare model and data
    is_classification = False
    if args.model == 'least':
        X = np.c_[X, np.ones((X.shape[0], 1))] if X.shape[1] == 2 else X
        model = LinearModel(X.shape[1])
        model_name = "Linear Regression"
    elif args.model == 'logistic':
        # create binary labels for logistic demo
        y_bin = (y > np.median(y)).astype(float)
        X = np.c_[X, np.ones((X.shape[0], 1))]
        model = LogisticModel(X.shape[1])
        y = y_bin
        is_classification = True
        model_name = "Logistic Regression"
    else:  # dnn
        model = DNNModel(in_size=3, hidden_size=16, out_size=1)
        model_name = "Deep Neural Network"

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, seed=42)
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Create updater based on optimizer choice
    if args.optimizer == 'sgd':
        updater = SGD(model.NumParameters(), X_train, y_train)
    else:  # svrg
        print("called SVRG") 
        updater = SVRG(model.NumParameters(), X_train, y_train)
    
    n_batches = (X_train.shape[0] + args.batch - 1) // args.batch
    print(f"Batches per epoch: {n_batches}")
    print(f"Starting training at {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)

    # Training loop
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        losses = []
       
        if args.optimizer == 'svrg':
            updater.EpochBegin(model)

        # Shuffle training data each epoch
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, X_train.shape[0], args.batch):
            xb = X_shuffled[i:i+args.batch]
            yb = y_shuffled[i:i+args.batch]
            
            try:
                grad = updater.Update(model, xb, yb)
                # apply gradient step
                model.para = model.para - args.lr * grad
                model.para = model.ProximalOperator(model.para, args.lr * 0.0)
                # compute loss on batch
                batch_loss = model.ComputeLoss(model.para, xb, yb)
                losses.append(batch_loss)
            except Exception as e:
                print(f"Warning: Error during training: {e}")
                print("Skipping this batch and continuing...")
                continue
        
        if not losses:
            print(f"Error: No successful batches in epoch {epoch+1}")
            break
            
        avg_loss = np.mean(losses)
        epoch_time = time.time() - epoch_start
        
        # Print progress every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(f'Epoch {epoch+1:2d}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.3f}s')
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("-" * 60)
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Average time per epoch: {training_time/args.epochs:.3f}s")
    
    # Compute test metrics
    test_loss = safe_loss(model.ComputeLoss(model.para, X_test, y_test))
    train_loss = safe_loss(model.ComputeLoss(model.para, X_train, y_train))
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Training Loss:   {train_loss:.6f}")
    print(f"Test Loss:       {test_loss:.6f}")
    
    # Compute and display accuracy or R² score
    try:
        if is_classification:
            train_acc = compute_accuracy(model, X_train, y_train)
            test_acc = compute_accuracy(model, X_test, y_test)
            print(f"Training Accuracy: {train_acc*100:.2f}%")
            print(f"Test Accuracy:     {test_acc*100:.2f}%")
        else:
            train_r2 = compute_r2_score(model, X_train, y_train)
            test_r2 = compute_r2_score(model, X_test, y_test)
            print(f"Training R² Score: {train_r2:.6f}")
            print(f"Test R² Score:     {test_r2:.6f}")
    except Exception as e:
        print(f"Note: Could not compute accuracy/R² metrics: {e}")
        print("Model training completed successfully, metrics computation skipped.")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
