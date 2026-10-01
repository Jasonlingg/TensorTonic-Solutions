import numpy as np

def select_by_index(arr: list, indices: list, axis: int) -> np.ndarray:
    """
    Returns a 2D float64 array of the selected rows or columns.
    """
    nparr = np.array(arr, dtype=float)

    if axis == 0:# return 2D float
        a = nparr[indices, :] 
    else:
        a = nparr[:,indices]

    return a
