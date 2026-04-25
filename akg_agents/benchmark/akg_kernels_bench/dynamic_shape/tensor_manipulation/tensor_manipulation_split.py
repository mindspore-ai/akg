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
    # Parameters for split
    split_size = 2048  # Size of each chunk
    dim = 1            # Dimension along which to split
    return [split_size, dim]