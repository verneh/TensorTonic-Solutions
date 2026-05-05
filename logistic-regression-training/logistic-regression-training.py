import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    N, D = X.shape
    
    # Initialize parameters
    w = np.zeros(D, dtype=float)
    b = 0.0
    
    # Gradient descent
    for _ in range(steps):
        # Forward pass: compute predictions
        z = X @ w + b  # Linear combination
        y_pred = _sigmoid(z)  # Predicted probabilities
        
        # Compute gradients
        # dL/dw = (1/N) * X^T * (y_pred - y)
        dw = (1.0 / N) * X.T @ (y_pred - y)
        # dL/db = (1/N) * sum(y_pred - y)
        db = (1.0 / N) * np.sum(y_pred - y)
        
        # Update parameters
        w = w - lr * dw
        b = b - lr * db
    
    return w, b