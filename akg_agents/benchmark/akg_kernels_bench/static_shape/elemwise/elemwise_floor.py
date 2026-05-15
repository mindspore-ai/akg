import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.floor(input, *, out=None)
        # Returns a new tensor with the floor of the elements of input, the largest integer less than or equal to each element.
        # This operation is commonly used in neural networks for:
        # - Quantization operations
        # - Discretizing continuous values
        # - Mathematical transformations
        return torch.floor(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for floor
    return []