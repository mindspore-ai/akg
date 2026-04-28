import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.tan(input, *, out=None)
        # Returns a new tensor with the tangent of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Implementing certain activation functions
        # - Mathematical transformations in specialized layers
        # - Periodic function approximations
        return torch.tan(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for tan
    return []