import numpy as np

def scale_cols(data: list, weights: list) -> np.ndarray:
    """
    Returns a float64 matrix with each column multiplied by its weight.
    """
    # unlike row scaling colum scaling does nto requrie a reshape becasue nump breacasint naturally aligns 1D arrays wit hlast dim of 2D arrays

    npdata = np.array(data, dtype=float)
    npweights = np.array(weights, dtype=float)

    return npdata * npweights
    
