import numpy as np

def angle_features(angles: list) -> np.ndarray:
    """
    Returns a (3, n) float64 array with sine, cosine, and tangent rows.
    """
    np_angles = np.array(angles, dtype=float)

    sin_angles = np.sin(np_angles)
    cos_angles = np.cos(np_angles)
    tan_angles = np.tan(np_angles)

    features = np.array([sin_angles, cos_angles, tan_angles])
    return features
