import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.squeeze(input, dim=None, out=None)
        # Returns a tensor with all the dimensions of input of size 1 removed.
        # This operation is commonly used in neural networks for:
        # - Removing unnecessary dimensions
        # - Cleaning up tensor shapes after operations
        # - Ensuring compatibility with other operations
        return torch.squeeze(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Squeezing removes the last dimension to get (1024, 4096)
    input_tensor = torch.randn(1024, 4096, 1, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for squeeze
    return []
