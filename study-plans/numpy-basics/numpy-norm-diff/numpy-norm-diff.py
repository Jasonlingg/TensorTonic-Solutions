import numpy as np

def norm_diff(a: list, b: list, lo: float, hi: float) -> np.ndarray:
    """
    Returns a float64 array of absolute normalized differences.
    """
    npa = np.array(a, dtype=float)
    npb = np.array(b, dtype=float)

    clippeda = np.clip(npa, lo, hi)
    clippedb = np.clip(npb, lo, hi)

    return abs(clippeda - clippedb) / (hi - lo)

