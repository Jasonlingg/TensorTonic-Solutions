import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    """
    Returns float64 correlation matrices for a, b, and their combined rows.
    """
    # compared relationship BETWEN COLS in two datsets and in in 
    # row wise combinaion

    a = np.asarray(a, dtype =float)
    b = np.asarray(b, dtype = float)

    # concat
    ab = np.concat([a,b], axis = 0)

    pearson_a = np.corrcoef(a, rowvar=False)
    pearson_b = np.corrcoef(b, rowvar=False)

    pearson = np.corrcoef(ab, rowvar=False)

    return np.stack([pearson_a, pearson_b, pearson])

