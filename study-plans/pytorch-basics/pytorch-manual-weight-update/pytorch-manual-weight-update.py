import torch
import torch.nn as nn

def manual_train_step(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module, lr: float) -> float:
    """
    Returns the pre-update batch loss as a Python float.
    """

    # given model, x data, y data, criterion (loss function), lr

    outputs = model(X)        
    loss = criterion(outputs,y)

    # backprop
    # Calculate gradients and store them in each parameter’s param.grad
    loss.backward()
    
    # update params
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    model.zero_grad()

    return loss.item()
