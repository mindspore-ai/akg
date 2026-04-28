import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, threshold=0.5):
        super(Model, self).__init__()
        self.threshold = threshold

    def forward(self, input_tensor):
        
        # Create condition: elements greater than threshold
        condition = input_tensor > self.threshold
        
        # Create alternative values (zeros)
        alt_values = torch.zeros_like(input_tensor)
        
        # Apply conditional selection
        return torch.where(condition, input_tensor, alt_values)


def get_inputs():
    input_tensor = torch.randn(256, 1024, 64, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for where
    return []
