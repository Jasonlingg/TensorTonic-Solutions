import numpy as np

def create_filled_array(shape: list, kind: str) -> np.ndarray:
    """
    Returns a 2D float64 array of zeros or ones with the requested shape.
    """
    if kind == "zeros":
        sol = np.zeros(shape,dtype = float)
    else:
        sol = np.ones(shape, dtype = float)
    return sol