import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.count_nonzero(input, dim=None)
        # Counts the number of non-zero values in the tensor input along the given dim.
        # If no dim is specified, count_nonzero() counts the total number of non-zero values in input.
        # This operation is commonly used in neural networks for:
        # - Sparsity analysis of weight tensors
        # - Computing activation statistics
        # - Implementing sparse operations
        return torch.count_nonzero(input_tensor, self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for counting
    # Count along second dimension (features dimension)
    dim = 1
    return [dim]