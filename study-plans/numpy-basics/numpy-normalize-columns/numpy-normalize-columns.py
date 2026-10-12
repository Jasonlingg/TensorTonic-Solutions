import numpy as np

def normalize(data: list) -> np.ndarray:
    """
    Returns a float64 matrix standardized independently by column.
    """

    npdata = np.array(data, dtype=float)

    # axiss 0 menas col
    mu = np.mean(npdata, axis = 0)
    sigma = np.std(npdata, axis = 0)

    return (npdata - mu) / sigma
