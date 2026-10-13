import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    # find eahc ROWs maximum ( so use axis =1 )
    # and minimum values and ther zeo bvase column indcies
    # when it occurs mroe than once use its first occurnece

    npdata = np.array(data, dtype = float)
    mx = np.max(npdata, axis=1)
    mn = np.min(npdata, axis=1)
    idxmx = np.argmax(npdata, axis =1)
    idxmn = np.argmin(npdata, axis =1)

    return np.stack([mx, idxmx, mn, idxmn])
    
    
