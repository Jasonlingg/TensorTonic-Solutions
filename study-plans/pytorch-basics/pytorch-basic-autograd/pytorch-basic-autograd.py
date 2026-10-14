import torch

def compute_gradient(values: torch.Tensor) -> torch.Tensor:
    """
    Returns a float32 gradient tensor with the same shape as values.
    """

    # enable gradient tracking on supplied tensor
    x = values.requires_grad_(True)

    # compute the scalar
    y = ((x ** 3) + 2 * x).sum()

    # backprop
    y.backward() 

    return x.grad

    
    
