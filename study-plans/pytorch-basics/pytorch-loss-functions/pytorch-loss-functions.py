import torch

def compute_loss(pred: torch.Tensor, target: torch.Tensor, method: str, delta: float = 1.0) -> float:
    """
    Returns the mean loss as a Python float.
    """
    if method == "mse":
        return ((pred - target)**2).mean().item()
    if method == "cross_entropy":
        # 1. Convert raw logits to log-probabilities safely
        log_probs = torch.log_softmax(pred, dim=1)
        # dim 1 is accross the colum axis note axis and dim are the same thing
        # 2. Extract the log-probability of the correct target class for each sample in the batch
        # We use a negative sign because probabilities are < 1, making their logs negative.
        loss_sample = -log_probs[range(len(target)), target]
        
        # 3. Average across the batch
        return loss_sample.mean().item()

    if method == "huber":
        # METHOD 1: The manual math way (using torch.where for conditional logic)
        
        # 1. Calculate the absolute difference between prediction and target
        error = torch.abs(pred - target)
        
        # 2. Calculate both possible mathematical states
        quadratic_loss = 0.5 * (error ** 2)
        linear_loss = delta * error - 0.5 * (delta ** 2)
        
        # 3. Apply the threshold: Use quadratic if error <= delta, else use linear
        loss_tensor = torch.where(error <= delta, quadratic_loss, linear_loss)
        
        return loss_tensor.mean().item()
        
        
