import torch

def reshape_tensor(x: torch.Tensor, op: str) -> torch.Tensor:
    """
    Returns the reshaped float32 tensor, including a scalar tensor after a complete squeeze.
    """
    if op == "flatten":
        return torch.flatten(x)
    elif op == "squeeze":
        return x.squeeze()
    elif op == "transpose":
        return torch.transpose(x,0,1)
