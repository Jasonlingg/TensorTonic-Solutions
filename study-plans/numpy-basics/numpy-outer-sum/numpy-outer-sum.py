import numpy as np

def outer_sum(a: list, b: list) -> np.ndarray:
    """
    Returns an (m, n) float64 array of pairwise sums.
    """
    anp = np.array(a, dtype=float)
    bnp = np.array(b, dtype=float)

    return np.add.outer(anp,bnp)
