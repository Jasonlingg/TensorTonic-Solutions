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
    # no gradient tracking because we dont need to track the changes
    # also no_grad is jsut a pytorch safeguard
    with torch.no_grad():
        # loop through each param in the model parameters
        # apply formaula z = z - lr*dz
        for param in model.parameters():
            param -= lr * param.grad

    model.zero_grad()

    return loss.item()
