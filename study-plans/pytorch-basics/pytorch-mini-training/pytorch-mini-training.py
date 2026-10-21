import torch
import torch.nn as nn

def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    """
    Returns the mean batch loss as a Python float.
    """
    # given model, dataloader, criterion, optimizer

    # First, we need to set train mode
    model.train()

    # intintalize loss
    total_loss = 0.0
    num_batches = 0

    for inputs, target in dataloader:
        
        # forward pass, take the test
        predictions = model(inputs)

        # calcualte loss
        loss = criterion(predictions, target)

        # gradient zeroing, we clear the memeory of the mistakes
        # on the last pass
        optimizer.zero_grad()

        
        # backward pass
        loss.backward()

        # now udpate the model wegihts usign calcualted gradients
        optimizer.step()

    
        # return mean batch loss
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
    
