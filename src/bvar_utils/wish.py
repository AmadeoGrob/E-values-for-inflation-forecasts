import numpy as np
from scipy.stats import wishart as _wishart  # private alias to keep the namespace clean


def wish(h, n, size=1, rng=None):
    """
    Sample W ~ Wishart_p(n, h) via SciPy.

    Parameters
    ----------
    h    : (p, p) array_like
           Symmetric positive-definite scale matrix.
    n    : int
           Degrees of freedom (must satisfy n ≥ p).
    size : int or tuple, optional
           Number of draws.  Default is 1 (a single matrix).
           If an int k is given, the result has shape (k, p, p).
    rng  : numpy.random.Generator or RandomState, optional
           Pseudorandom-number generator for reproducibility.

    Returns
    -------
    W : ndarray
        * A single p×p matrix if size == 1 (default);
        * an array of shape (k, p, p) if size == k.

    Note:
    -----
    The parameterisation matches MATLAB:  **E[W] = n·h**.
    Internally we call ``scipy.stats.wishart.rvs`` which performs the same
    Bartlett decomposition but in compiled code for speed and robustness.
    """
    h = np.asarray(h)
    if h.shape[0] != h.shape[1]:
        raise ValueError("Scale matrix 'h' must be square.")
    p = h.shape[0]
    if n < p:
        raise ValueError(f"Degrees of freedom n must be ≥ p={p}.")
    if rng is None:
        rng = np.random.default_rng()

    # SciPy does the Bartlett decomposition internally
    W = _wishart.rvs(df=n, scale=h, size=size, random_state=rng)

    # collapse shape when size=1
    if np.isscalar(size) or size == 1:
        return np.asarray(W)  # returns (p, p)
    return W  # returns (size, p, p)
