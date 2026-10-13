import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    npdata = np.array(data, dtype = float)

    return np.stack([np.mean(npdata, axis = axis), 
                     np.std(npdata, axis= axis), np.min(npdata, axis=axis), np.max(npdata, axis=axis), ])

    
    
