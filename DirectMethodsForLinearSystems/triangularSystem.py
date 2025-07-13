import numpy as np

# Solves a lower triangular system Lx = b using forward substitution
def forward_substitution(L, b):
    n = len(b)
    x = np.zeros_like(b, dtype=float)  # Initialize solution vector x with zeros

    for i in range(n):
        if L[i, i] == 0:
            raise ValueError("Matrix is singular!")  # Cannot divide by zero
        # Compute x[i] using previously computed x[0] to x[i-1]
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    return x

# Solves an upper triangular system Ux = b using backward substitution
def backward_substitution(U, b):
    n = len(b)
    x = np.zeros_like(b, dtype=float)  # Initialize solution vector x with zeros

    for i in reversed(range(n)):  # Start from the last row and go upward
        if U[i, i] == 0:
            raise ValueError("Matrix is singular!")  # Cannot divide by zero
        # Compute x[i] using previously computed x[i+1] to x[n-1]
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x
