import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Pad or truncate sequences to equal length.
    
    Parameters
    ----------
    seqs : list of lists
        List of token ID sequences (each sequence is list of ints)
    max_len : int or None, default=None
        Target length for all sequences. If None, use max length from seqs
    pad_value : int, default=0
        Value to use for padding
    Returns
    -------
    numpy.ndarray, shape (N, L) where L = max_len (if provided) or max sequence length
        Padded sequences as int array
    """
    # Handle empty input
    if not seqs:
        return np.zeros((0, 0), dtype=int)
    
    N = len(seqs)
    
    # Determine max length
    if max_len is None:
        # Find maximum sequence length
        max_len = max(len(seq) for seq in seqs)
    
    # Initialize output array with pad_value
    padded = np.full((N, max_len), pad_value, dtype=int)
    
    # Fill sequences (truncating if needed)
    for i, seq in enumerate(seqs):
        # Take up to max_len elements (truncation at the end)
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
    
    return padded
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    pass