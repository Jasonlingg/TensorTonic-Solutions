import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    # turn into numpy array
    npdata = np.array(data, dtype=float)
    if npdata is None:
        return null

    # given  a 2D pyhton list
    # 3layers in thsi order
    # 1. 1.0 if value is > than threshold, otherwise replace with zeors
    mask_1 = np.where(npdata > threshold, 1.0, 0.0)
    # 2. Any rolw filter: if value> threshold, do nothing; otherwise, replace with zeros
    cond1 = np.any(npdata > threshold, axis =1, keepdims=True)
    mask_2 = np.where( cond1, npdata, 0.0)
    # 3. All row filter: if value> threshold, do nothing; otherwise,
    # replace entire row with zeros
    cond2 = np.all(npdata > threshold, axis=1, keepdims=True)
    mask_3 = np.where(cond2, npdata, 0.0)

    return np.stack((mask_1, mask_2, mask_3))


    
    
