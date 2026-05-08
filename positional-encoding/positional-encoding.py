import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Create position indices: (seq_len, 1)
    pos = np.arange(seq_len)[:, np.newaxis]
    
    # Create dimension indices: (d_model)
    i = np.arange(d_model)
    
    # Compute the frequencies for each dimension
    # For even indices: use base^(-2i/d_model), for odd: base^(-2i/d_model) as well
    # but sin/cos will be applied alternately
    # Create angle_rates: shape (d_model)
    angle_rates = 1.0 / np.power(base, (2 * (i // 2)) / d_model)
    
    # Compute angles: shape (seq_len, d_model)
    angles = pos * angle_rates
    
    # Apply sin to even indices (0, 2, 4, ...) and cos to odd indices (1, 3, 5, ...)
    # For odd d_model, the last column will be sin
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Even indices: sin
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # Odd indices: cos
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe