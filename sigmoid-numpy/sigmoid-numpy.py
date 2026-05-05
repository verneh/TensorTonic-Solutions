import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Convert input to numpy array (handles scalars and lists)
    x = np.asarray(x, dtype=float)
    
    # Write code here
    return 1 / (1 + np.exp(-x))


sigmoid([0, 2, -2])