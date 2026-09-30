import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    npdata = np.array(data, dtype=float)
    if operation == "flatten":
        sol = npdata.flatten()
    elif operation == "add_batch":
        sol = np.expand_dims(npdata, axis=0)
    else:
        sol = np.transpose(npdata)

    return sol