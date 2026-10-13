import numpy as np

def tile_diff(data: list, reps: int) -> np.ndarray:
    """
    Returns float64 slices of tiled values and next-row differences with a final zero row.
    """
    # tiling replicates array along speciifeid axes
    # np.tile(A, reps) contstas an array by repeating A (col, row)
    # 

    data = np.asarray(data, dtype = float)

    tiled = np.tile(data, (reps, 1))
    diff = np.diff(tiled, axis= 0)

    padded_diff = np.pad(diff, ((0, 1), (0, 0)), mode='constant', constant_values=0)
    return np.stack([tiled, padded_diff])

