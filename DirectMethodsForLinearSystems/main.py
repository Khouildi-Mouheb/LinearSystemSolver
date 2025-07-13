import numpy as np
from DirectMethodsForLinearSystems.solveCholesky import solve_cholesky  # Import Cholesky-based solver

# Entry point of the program
if __name__ == "__main__":
    try:
        # Define a symmetric, positive definite matrix A
        A = np.array([[4, 12, -16],
                      [12, 37, -43],
                      [-16, -43, 98]], dtype=float)

        # Define the right-hand side vector b
        b = np.array([1, 2, 3], dtype=float)

        print("Solving Ax = b using Cholesky decomposition:")
        print("A =\n", A)
        print("b =", b)

        # Solve the system using Cholesky
        x = solve_cholesky(A, b)

        # Display the solution
        print("\nSolution x =", x)

        # Optional: verify the solution by computing A @ x
        print("\nCheck: A @ x =", np.dot(A, x))
        print("Original b =", b)

    # Handle any error (e.g. if matrix is not positive definite)
    except Exception as e:
        print("Error occurred:", e)
