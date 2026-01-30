import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.rsqrt(input, *, out=None)
        # Returns a new tensor with the reciprocal square root of the elements of input.
        # rsqrt(input) = 1 / sqrt(input)
        # This operation is commonly used in neural networks for:
        # - Normalization operations (e.g., RMS normalization, layer normalization)
        # - Mathematical transformations in specialized layers
        # - Implementing certain activation functions
        return torch.rsqrt(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Using positive values for rsqrt since sqrt of negative numbers is not defined in reals
    input_tensor = torch.rand(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to avoid division by zero
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for rsqrt
    return []