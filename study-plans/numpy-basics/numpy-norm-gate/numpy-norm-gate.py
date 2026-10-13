import numpy as np

def norm_gate(X: list, W: list, threshold: float) -> np.ndarray:
    """
    Returns an (n, k) float64 matrix of norm-gated transformed rows.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)

    Y = X @ W

    # note Y is n, m
    
    # comptue teh L2 norm
    norm = np.sqrt(np.sum(Y**2, axis = 1))
    gate = norm >= threshold
    # gate has the shape of (n, )
    # we need it to broadcast onto Y
    Z = Y * gate.reshape(-1,1)
    return Z
    
    

    
    
