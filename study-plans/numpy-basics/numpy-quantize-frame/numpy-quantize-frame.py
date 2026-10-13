import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    # 3 versions of a rectangular python list
    # nearest rounding
    # floor to int, 
    # cleing to int

    arr = np.asarray(data, dtype=np.float64)
    
    # 1. Calculate the arrays upfront 
    transformations = [
        np.round(arr, decimals),
        np.floor(arr),
        np.ceil(arr)
    ]
    
    results = []
    
    # 2. Loop through the pre-calculated arrays to apply the padding
    for transformed in transformations:
        padded = np.pad(transformed, pad_width, mode='constant', constant_values=0)
        results.append(padded)
        
    # 3. Convert the Python list of arrays into a single 3D NumPy array
    return np.array(results, dtype=np.float64)
    
