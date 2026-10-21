import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """

    # given microbatches, and the leanring rate lr and the accum_steps


    # need to require grad to be set true
    w = w_init.clone().detach().to(torch.float32).requires_grad_(True)
    last_avr_grad = None


    # batche includes input and output 
    # the "i" here is for 
    for i, (data, targets) in enumerate(micro_batches):
        
        predictions = w.T @ data

        loss = (predictions - targets)**2

        # now we have the loss we can compute gradients for it
        loss.backward()

        # after computign gradient, we check if we need to step
        # since enumberate counts form 0 we need to add 1
        # we take a step when we reach the end of batch or the size of batch
        if (i + 1) % accum_steps == 0 or (i + 1) == len(micro_batches):
            last_avr_grad = w.grad.clone() / accum_steps

            # Using grad is essentially camera crew following what is happening underneath
            # we use with to set that indented block to not track grad
            with torch.no_grad():
                # remember from class that w_new = w_old - lr * dl/dw
                # also remember that getting the gradient rqurei funciton and since w had requires_grad_(True)
                # we jsut need to get htat attribute
                w -= lr * last_avr_grad

            # reset the gradients for the next batch
            w.grad.zero_()

    return w.detach(), last_avr_grad.detach()
                
        

        

        
