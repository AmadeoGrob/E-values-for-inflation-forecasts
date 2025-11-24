import numpy as np


def mlag2(X, p):
    """
    Create a matrix of p lags of X, stacked side-by-side.

    Parameters
    ----------
    X : (T, N) array_like
        Time-ordered data, T observations × N series.
    p : int
        Highest lag to include (p ≥ 1).

    Returns
    -------
    Xlag : (T, N*p) ndarray
        For each t > p-1, the block
            Xlag[t, N*(i-1):N*i] = X[t-i, :]
        for i = 1, …, p.
        The first p rows are zeros.

    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (T observations × N series).")
    if p < 1 or not int(p) == p:
        raise ValueError("p must be a positive integer.")

    T, N = X.shape
    Xlag = np.zeros((T, N * p), dtype=X.dtype)

    # Bartlett-style loop
    for i in range(1, p + 1):
        Xlag[p:, (i - 1) * N : i * N] = X[p - i : T - i, :]

    return Xlag
