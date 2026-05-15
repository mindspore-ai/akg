import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.flip(input, dims)
        # Reverses the order of elements along the given dimension(s).
        # This operation is commonly used for:
        # - Data augmentation
        # - Sequence reversal
        # - Image flipping
        # - Creating symmetric patterns
        
        # Flip along specified dimensions (changed to dim=0 for batch dimension)
        flipped = torch.flip(input_tensor, dims=(0,))
        
        # Return the flipped tensor
        return flipped


def get_inputs():
    # Batch size: 1024 (larger batch)
    # Hidden dimension: 4096 (larger hidden dim)
    input_tensor = torch.randn(2, 256, 16, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for flip - now flipping along dim=0 (batch dimension)
    return []
