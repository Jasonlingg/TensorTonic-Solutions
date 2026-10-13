import numpy as np

def winsorize(data: list, lo_q: float, hi_q: float) -> np.ndarray:
    """
    Returns float64 slices of clipped values, lower mask, and upper mask.
    """
    npdata = np.asarray(data, dtype = float)

    #  get the lower percentile 
    # since we are limiting outlier in each column to the percentiles, we use axis = 0
    lo = np.percentile(npdata, lo_q, axis=0)
    hi = np.percentile(npdata, hi_q, axis = 0)

    # after figuing out the vlaue of the lower and upper bound
    # go ahead and clip
    clipped = np.clip(npdata, lo, hi)
    lower_mask = (npdata < lo)
    upper_mask = (npdata > hi)
    return np.stack([clipped, lower_mask,upper_mask])
    
