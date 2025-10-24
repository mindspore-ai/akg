import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Split operation using torch.chunk for equal-sized split_size.
    This operation is commonly used in neural networks for:
    - Creating equal-sized split_size for parallel processing
    - Implementing multi-head attention with equal head sizes
    - Used in architectures that require balanced tensor partitioning
    """
    def __init__(self, split_size=None, dim=None):
        super(Model, self).__init__()
        self.split_size = split_size
        self.dim = dim

    def forward(self, input_tensor):
        # torch.chunk(tensor, split_size, dim=2)
        # Splits the tensor into equal-sized split_size along the specified dimension
        return torch.chunk(input_tensor, split_size=self.split_size, dim=self.dim)


def get_inputs():
    # Splitting along channel dimension with split_size 511 gives us 7 chunks of (16, 1024, 511) and one chunk of (16, 1024, 6)
    input_tensor = torch.randn(16, 1024, 3072, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters for chunk operation
    split_size = 511  # Number of equal-sized split_size to create
    dim = 2     # Dimension along which to split (hidden dimension)
    return [split_size, dim]
