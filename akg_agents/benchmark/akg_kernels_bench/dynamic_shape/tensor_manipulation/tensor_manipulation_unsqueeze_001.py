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


def get_inputs_dyn_list():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Three tensors are concatenated along the batch dimension to get (1536, 4096)
    # Small shape case
    inputs1 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case
    inputs2 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case
    inputs3 = torch.randn(512, 4096, dtype=torch.float32)
    # Standard Large shape case
    inputs4 = torch.randn(1024, 4096, dtype=torch.float32)
    # Large shape case
    inputs5 = torch.randn(2048, 8192, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5]
    ]


def get_init_inputs():
    # Specific dim value for unsqueeze
    dim = 1  # Insert dimension at position 1
    return [dim]