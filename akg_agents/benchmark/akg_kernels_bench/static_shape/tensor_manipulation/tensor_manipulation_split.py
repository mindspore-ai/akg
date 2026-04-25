import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, split_size=None, dim=None):
        super(Model, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, input_tensor):
        # torch.split(tensor, split_size_or_sections, dim=0)
        # Splits the tensor into chunks. Each chunk is a view of the original tensor.
        # This operation is commonly used in neural networks for:
        # - Dividing feature maps into multiple parts
        # - Implementing parallel processing paths
        # - Splitting tensors for different heads in multi-head attention
        return torch.split(input_tensor, self.split_size, dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Splitting along dimension 1 with split_size 2048 gives us two chunks of (1024, 2048)
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for split
    split_size = 2048  # Size of each chunk
    dim = 1            # Dimension along which to split
    return [split_size, dim]