import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, out_features: int):
        super().__init__()
        # 3 components
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, out_features)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor of shape (batch, out_features).
        """
        x = self.linear1(x)
        x = self.relu(x)
        x= self.linear2(x)
        return x        
