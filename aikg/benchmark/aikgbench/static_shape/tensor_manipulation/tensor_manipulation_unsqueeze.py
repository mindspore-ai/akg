import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.unsqueeze(input, dim, out=None)
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        # This operation is commonly used in neural networks for:
        # - Adding dimensions to tensors for compatibility
        # - Preparing tensors for broadcasting
        # - Implementing certain operations that require specific dimensions
        return torch.unsqueeze(input_tensor, -1)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Unsqueezing adds a new dimension at the end to get (1024, 4096, 1)
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for unsqueeze
    return []