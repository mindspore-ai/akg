import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.unsqueeze(input, dim, out=None)
        # Returns a new tensor with a dimension of size one inserted at the specified position.
        # This operation is commonly used in neural networks for:
        # - Adding dimensions to tensors for compatibility with other operations
        # - Preparing tensors for broadcasting
        # - Implementing certain operations that require specific dimensions
        return torch.unsqueeze(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 256
    # Sequence length: 64
    # Hidden dimension: 512
    # Unsqueezing adds a new dimension at position -2 (second to last) to get (4096, 1, 250, 512)
    input_tensor = torch.randn(4096, 250, 512, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for unsqueeze
    dim = -2  # Insert dimension at second to last position
    return [dim]
