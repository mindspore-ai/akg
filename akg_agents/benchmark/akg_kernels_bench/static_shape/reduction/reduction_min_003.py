import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.min(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the minimum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each minimum value found.
        # This operation is commonly used in neural networks for:
        # - Finding the least activated neuron in a layer
        # - Implementing min-pooling operations
        # - Computing robust statistics in normalization layers
        return torch.min(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for global min
    return []