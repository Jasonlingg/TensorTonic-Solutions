import torch
import math
import torch.nn as nn

class CustomLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # first need to compute raw weight 
        # parameters must be nn.parameters so they register
        self.weight = nn.Parameter(torch.empty(out_features,in_features, dtype = torch.float32))

        # kaiming He Initialization
        # worries only about fan-in weights, means we get issues with the backward pass
        torch.nn.init.kaiming_uniform_(self. weight, a= math.sqrt(5))

        # can just set bias to zero
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        return x @ self.weight.T + self.bias