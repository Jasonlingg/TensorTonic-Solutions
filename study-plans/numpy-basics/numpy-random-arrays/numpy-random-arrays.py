import numpy as np

def generate_random_array(shape: list, kind: str, seed: int) -> np.ndarray:
    """
    Returns a seeded 2D float64 random array with the requested shape.
    """
    shaper = tuple(shape)
    rng = np.random.default_rng(seed=seed)
    if kind == "normal":
        x =rng.standard_normal(shaper)
    else:
        x= rng.random(shaper)

    return x
    
