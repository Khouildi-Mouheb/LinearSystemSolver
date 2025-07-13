from DirectMethodsForLinearSystems.choleskyFactorisation import cholesky_decomposition
from DirectMethodsForLinearSystems.triangularSystem import forward_substitution, backward_substitution

# Solves the system Ax = b using Cholesky decomposition
# Assumes A is symmetric and positive definite
def solve_cholesky(A, b):
    # Step 1: Decompose A into L such that A = L @ L.T
    L = cholesky_decomposition(A)
    
    # Display the lower triangular matrix and its transpose
    print("L:\n", L)
    print("L.T (transpose of L):\n", L.T)

    # Step 2: Solve Ly = b using forward substitution
    y = forward_substitution(L, b)

    # Step 3: Solve Lᵗx = y using backward substitution
    x = backward_substitution(L.T, y)

    return x  # Final solution to Ax = b
