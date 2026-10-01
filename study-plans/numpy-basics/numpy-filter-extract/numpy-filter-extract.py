import numpy as np

def filter_and_extract(data: list, row_start: int, row_stop: int, threshold: float) -> np.ndarray:
    """
    Returns matching values in row-major order as a 1D float64 array.
    """
    npdata = np.array(data, dtype=float)

    # selct rows betwen start and stop
    # kee values > threshold
    # retunr mathcing values as a arrray

    sliceddata = npdata[row_start: row_stop]


    mask = sliceddata > threshold
    return sliceddata[mask]    
