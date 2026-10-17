import torch

def batch_norm(X: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns a float32 tensor with the same shape as X.
    """
    # we are given gamma beta and eps
    mean = torch.mean(X,dim=0)

    variance = torch.var(X, dim=0, unbiased=False)

    return (gamma * (X - mean)/(torch.sqrt(variance + eps)) + beta)
    
