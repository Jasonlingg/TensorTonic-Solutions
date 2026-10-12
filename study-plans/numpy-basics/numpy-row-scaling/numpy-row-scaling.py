import numpy as np

def scale_rows(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each row multiplied by its weight.
    """
    # data
    npdata = np.array(data, dtype=float)

    # need to reshape
    npweights = np.array(weights, dtype=float)

    return npdata * npweights.reshape(-1,1)

    
