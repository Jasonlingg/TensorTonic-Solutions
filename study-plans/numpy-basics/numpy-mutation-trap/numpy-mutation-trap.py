import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    # selct row from a given index
    # preserve its orgiina values and produce a clipped verison
    # vlaues inside remind unchanged

    npdata = np.array(data, dtype=float)
    row = npdata[row_idx, :]
    orginal_row = row.copy()
    clipped_row = np.clip(row, lo, hi, out=row)

    return np.stack((orginal_row, clipped_row))