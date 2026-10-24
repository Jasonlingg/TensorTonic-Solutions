import torch
import torch.nn as nn

def train_with_early_stopping(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, max_epochs: int, patience: int) -> dict:
    """
    Returns train_losses and val_losses as float lists, and stopped_epoch as an int.
    """
    # Given:
    # model, train_loader, val_loader, criterion, optimizer, max_epoch, patience
    # basically type of regularization, patience is the # of epochs 
    # where we dont see an imrpvoemtn of valdiation loss when increaing trainging

    # we need store training and validation loss
    loss_train = []
    loss_val = []
    best_val_loss = float("inf")
    stopped_epoch = 0
    epochs_without_improvement = 0

    # for loop for epochs
    for i in range(1, max_epochs + 1):
        # turn on train
        model.train()
        # we need  to calculate the average of the gradients 
        training_loss = 0.0
        training_examples = 0

        # now load the data for train
        # note that the loaders give batche sizes
        # doign SGD so we update at teh end of every batch
        for inputs, targets in train_loader:
            
            # clear grads and loss accumulator
            optimizer.zero_grad()
            loss = 0.0
            
            # forward
            pred = model(inputs)
            # loss
            loss = criterion(pred, targets)
            # backprop
            loss.backward()
            # SGD so we update with optimizer.step()
            optimizer.step()
            # save loss
            training_loss += loss.item()
            training_examples += 1

        # done all the batches so we add that to our list of
        # losses for each epoch
        loss_train.append(training_loss / training_examples)

        # validation is a form of eval so 
        model.eval()

        # reset val loss
        val_loss = 0.0
        val_examples = 0

        # now load the dat for validation, we need no_grad (important!)
        with torch.no_grad():
            for inputs, targets in val_loader:
                # no optimizer we just calcualte loss
                pred = model(inputs)
                loss = criterion(pred, targets)

                # now we track loss with accumulator
                val_loss += loss.item()
                val_examples += 1
                

        # append the total losee from all the batches to the loss list
        loss_val.append(val_loss / val_examples)
        average_val_loss= val_loss / val_examples
    
        # now we need to check if we havea new best val loss
        if best_val_loss > average_val_loss:
            best_val_loss = average_val_loss
            # found a loss that is lower so keep training 
            # reset this
            epochs_without_improvement = 0
        else:
            # training more did not improve the model
            epochs_without_improvement += 1

        stopped_epoch = i


        if epochs_without_improvement >= patience:
            break

    return {
        "train_losses": loss_train,
        "val_losses": loss_val,
        "stopped_epoch": stopped_epoch,
    }
            
            
            
        
    
