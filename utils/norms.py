import numpy as np

def norm_1(x):
    """L1 norm (sum of absolute values)"""
    return np.sum(np.abs(x))

def norm_2(x):
    """L2 norm (Euclidean norm)"""
    return np.sqrt(np.sum(x**2))

def norm_inf(x):
    """Infinity norm (maximum absolute value)"""
    return np.max(np.abs(x))

def norm_holder(x, p):
    """
    Hölder (p-norm)
    p must be >= 1
    """
    if p < 1:
        raise ValueError("p must be >= 1 for Hölder norm.")
    return np.power(np.sum(np.abs(x) ** p), 1/p)

def relative_error(x_exact, x_approx, norm_type='2', p=None):
    """
    Computes the relative error between exact and approximate vectors.

    Parameters:
    - norm_type: '1', '2', 'inf', or 'holder'
    - p: used only for Hölder norm

    Returns:
    - Relative error value
    """
    diff = x_exact - x_approx

    if norm_type == '1':
        return norm_1(diff) / norm_1(x_exact)
    elif norm_type == '2':
        return norm_2(diff) / norm_2(x_exact)
    elif norm_type == 'inf':
        return norm_inf(diff) / norm_inf(x_exact)
    elif norm_type == 'holder':
        if p is None:
            raise ValueError("Please specify a value for p in Hölder norm.")
        return norm_holder(diff, p) / norm_holder(x_exact, p)
    else:
        raise ValueError("Unsupported norm type. Choose from '1', '2', 'inf', or 'holder'.")
