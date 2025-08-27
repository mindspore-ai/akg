import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, exponent):
        # torch.pow(input, exponent, *, out=None)
        # Takes the power of each element in input with exponent and returns a tensor with the result.
        # This is a power operation with exponent=0.5 (square root).
        # Power operations are commonly used in neural networks for:
        # - Implementing polynomial activation functions
        # - Computing distance metrics
        # - Mathematical transformations in specialized layers
        return torch.pow(input_tensor, exponent)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.rand(1024, 4096, dtype=torch.float32) + 1e-6  # Using positive values for square root
    exponent = torch.full((1024, 4096), 0.5, dtype=torch.float32)  # Square root operation
    return [input_tensor, exponent]


def get_init_inputs():
    # No parameters needed for pow
    return []