import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # torch.floor_divide(input, other, *, out=None)
        # Computes input divided by other element-wise, rounded down to the nearest integer.
        # This operation is commonly used in neural networks for:
        # - Quantization operations
        # - Discretizing continuous values
        # - Mathematical transformations
        return torch.floor_divide(dividend, divisor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    dividend = torch.randn(1024, 4096, dtype=torch.float32)
    divisor = torch.randn(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to avoid division by zero
    return [dividend, divisor]


def get_init_inputs():
    # No parameters needed for floor_divide
    return []