import numpy as np

def pairwise_diff(a: list) -> np.ndarray:
    """
    Returns an (n, n) float64 array of signed pairwise differences.
    """

    anp = np.array(a, dtype=float)

    return anp.reshape(-1, 1) - anp.reshape(1,-1)
