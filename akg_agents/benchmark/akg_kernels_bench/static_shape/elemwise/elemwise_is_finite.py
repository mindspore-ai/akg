import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.isfinite(input, *, out=None)
        # Returns a new tensor with boolean elements representing if each element of input is finite or not.
        # This operation is commonly used in neural networks for:
        # - Detecting invalid values in tensors
        # - Implementing data validation checks
        # - Creating masks for valid/invalid data
        return torch.isfinite(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for isfinite
    return []