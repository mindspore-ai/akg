import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
        # Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        # This operation is commonly used in neural networks for:
        # - Creating evenly spaced values for interpolation
        # - Generating sequences with specific spacing
        # - Implementing certain mathematical transformations
        return torch.linspace(0.0, 1.0, 4096, dtype=torch.float32)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for linspace
    return []