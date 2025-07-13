import numpy as np
from DirectMethodsForLinearSystems.triangularSystem import forward_substitution, backward_substitution

# Performs Cholesky decomposition of a symmetric, positive definite matrix A
# Returns lower triangular matrix L such that A = L @ L.T
def cholesky_decomposition(A):
    n = A.shape[0]                     # Number of rows (assumes A is square)
    L = np.zeros_like(A)              # Initialize L as a zero matrix (same shape as A)

    for i in range(n):
        for j in range(i + 1):        # Only compute lower triangle and diagonal (j ≤ i)
            sum_val = np.dot(L[i, :j], L[j, :j])  # Compute the sum of products for dot product
            
            if i == j:
                # Diagonal element: sqrt(A[i, i] - sum)
                val = A[i, i] - sum_val
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(val)
            else:
                # Off-diagonal element: compute using already computed entries of L
                L[i, j] = (A[i, j] - sum_val) / L[j, j]
    
    return L  # Lower triangular matrix such that A = L @ L.T
