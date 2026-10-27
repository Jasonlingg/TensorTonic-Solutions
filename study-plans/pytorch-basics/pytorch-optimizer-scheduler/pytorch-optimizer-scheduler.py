import torch
import torch.nn as nn

def train_with_scheduler(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.StepLR, num_epochs: int) -> dict:
    """
    Returns losses and lrs as lists of Python floats in a dictionary.
    """

    # return losses and learning rates as lists of python floats

    losses = []
    lrs = []

    # we are gviein model, dataloader, cirterion (loss), optizmier
    # schedular and num epoches
    
    for epoch in range(num_epochs):
        model.train()
        # to get te current lr get the first layer lr param
        curr_lr = optimizer.param_groups[0]['lr']
        lrs.append(curr_lr)
        # start training loop for each batch
        total_loss = 0.0
        total_batches = 0

        for input, target in dataloader:
            optimizer.zero_grad()
            prediction = model(input)
            loss = criterion(prediction,target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1

        losses.append(total_loss/ total_batches)
        scheduler.step()

    return {"losses": losses, "lrs": lrs}

            
            
            
            
    
