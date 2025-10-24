import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Split operation along dimension 0 (batch dimension).
    This operation is commonly used in neural networks for:
    - Dividing batch data for parallel processing
    - Implementing data parallelism
    - Splitting sequences for different processing paths
    """
    def __init__(self, split_size=None, dim=None):
        super(Model, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, input_tensor):
        # torch.split(tensor, split_size_or_sections, dim=0)
        # Splits the tensor into chunks along dimension 0 (batch dimension)
        return torch.split(input_tensor, self.split_size, dim=self.dim)


def get_inputs():
    # Splitting along dimension 0 with split_size 32 gives us two chunks of (32, 512, 2048)
    input_tensor = torch.randn(64, 512, 2048, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for split
    split_size = 32  # Size of each chunk along batch dimension
    dim = 0          # Dimension along which to split (batch dimension)
    return [split_size, dim]
