import numpy as np
from DirectMethodsForLinearSystems.triangularSystem import forward_substitution, backward_substitution

# Performs LU decomposition of matrix A (without pivoting)
# Returns lower triangular matrix L and upper triangular matrix U such that A = LU
def lu_decomposition(A):
    A = A.astype(float)          # Ensure A is in float to avoid integer division
    n = A.shape[0]               # Get the number of rows (assumes A is square)
    L = np.eye(n)                # Initialize L as the identity matrix
    U = A.copy()                 # Copy A to U (will be transformed into upper triangular)

    for i in range(n):
        if U[i, i] == 0:
            raise ValueError("Zero pivot encountered. Use pivoting.")  # Avoid division by zero
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]  # Compute the multiplier to eliminate below the pivot
            L[j, i] = factor            # Store the multiplier in L
            U[j, i:] -= factor * U[i, i:]  # Eliminate entries below the pivot
    return L, U
