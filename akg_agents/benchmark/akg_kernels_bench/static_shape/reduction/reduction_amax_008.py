import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.max(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the maximum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each maximum value found.
        return torch.amax(input_tensor, self.dim)


def get_inputs():
    # Batch size: 16
    # Hidden dimension: 32
    # Sequence length: 2048
    input_tensor = torch.randn(16, 32, 2048, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Reduce along first and second dimension
    dim = [0, 1]
    return [dim]