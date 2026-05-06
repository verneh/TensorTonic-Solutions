import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    N = len(A)  # number of rows
    M = len(A[0]) if N > 0 else 0  # number of columns
    
    # Create result array
    result = np.zeros((M, N))
    
    # Transpose using manual indexing
    for i in range(N):
        for j in range(M):
            result[j, i] = A[i][j]
    
    return result
