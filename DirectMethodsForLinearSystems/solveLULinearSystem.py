from DirectMethodsForLinearSystems.LUDecomposition import lu_decomposition
import numpy as np
from DirectMethodsForLinearSystems.triangularSystem import forward_substitution, backward_substitution

# Solves the linear system Ax = b using LU decomposition
def solve_LU_linear_system(A, b):
    L, U = lu_decomposition(A)  # Decompose A into L and U
    print("Matrix L:\n", L)     # Display L
    print("Matrix U:\n", U)     # Display U

    y = forward_substitution(L, b)  # Solve Ly = b using forward substitution
    x = backward_substitution(U, y)  # Solve Ux = y using backward substitution
    return x
