import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Split operation with smaller chunk size for fine-grained processing.
    This operation is commonly used in neural networks for:
    - Fine-grained feature processing
    - Implementing attention mechanisms with smaller chunks
    - Memory-efficient processing of large tensors
    """
    def __init__(self, split_size=None, dim=None):
        super(Model, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, input_tensor):
        # torch.split(tensor, split_size_or_sections, dim=1)
        # Splits the tensor into smaller chunks for fine-grained processing
        return torch.split(input_tensor, self.split_size, dim=self.dim)


def get_inputs():
    # Splitting along dimension 1 with split_size 256 gives us 8 chunks of (128, 256, 1024)
    input_tensor = torch.randn(128, 2048, 1024, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for split
    split_size = 256  # Smaller chunk size for fine-grained processing
    dim = 1           # Dimension along which to split (sequence dimension)
    return [split_size, dim]
