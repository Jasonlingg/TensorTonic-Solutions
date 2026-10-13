import numpy as np

def sort_with_indices(data: list, axis: int) -> np.ndarray:
    """
    Returns a (2, m, n) float64 array of sorted values and source indices.
    """
    # sort regulaur list in ascending order, reprot each valeus original pos

    npdata = np.array(data, dtype = float)

    
    sorted = np.sort(npdata, axis =axis)
    idx = np.argsort(npdata, axis =axis)

    return np.stack([sorted, idx])
    
    
