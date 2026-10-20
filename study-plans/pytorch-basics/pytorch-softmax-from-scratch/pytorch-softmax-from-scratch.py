import torch

def softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 probability tensor with the same shape as logits.
    """

    max_vals = torch.max(logits, dim=1, keepdim= True).values
    max_subtract = logits - max_vals

    exp = torch.exp(max_subtract)

    return exp / torch.sum(exp, dim=1, keepdims=True)
    
