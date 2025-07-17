import numpy as np

def eigenvalues(A):
    """
    Returns the eigenvalues of a square matrix A.

    Parameters:
        A : ndarray
            A square matrix (n x n)

    Returns:
        ndarray: Array of eigenvalues (possibly complex)
    """
    return np.linalg.eigvals(A)


def spectral_radius(A):
    """
    Computes the spectral radius of a square matrix A.

    Parameters:
        A : ndarray
            A square matrix (n x n)

    Returns:
        float: The spectral radius (maximum absolute value of eigenvalues)
    """
    eigs = eigenvalues(A)
    return np.max(np.abs(eigs))

