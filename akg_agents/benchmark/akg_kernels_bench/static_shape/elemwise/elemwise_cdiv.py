import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # torch.div(input, other, *, rounding_mode='trunc', out=None)
        # Performs division with truncation towards zero.
        # This is equivalent to C-style integer division.
        # This operation is commonly used in neural networks for:
        # - Implementing ceiling division operations
        # - Calculating grid dimensions in CUDA kernels
        # - Mathematical transformations that require integer division
        return torch.div(dividend, divisor, rounding_mode='trunc')


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    dividend = torch.randn(1024, 4096, dtype=torch.float32)
    divisor = torch.randn(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to avoid division by zero
    return [dividend, divisor]


def get_init_inputs():
    # No parameters needed for cdiv
    return []