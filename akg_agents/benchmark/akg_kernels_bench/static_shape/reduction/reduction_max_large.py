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
        # This operation is commonly used in neural networks for:
        # - Max pooling in convolutional networks
        # - Finding the most activated neuron in a layer
        # - Attention mechanisms in transformers
        return torch.max(input_tensor, self.dim)


def get_inputs():
    # Batch size: 2048
    # Hidden dimension: 8192
    input_tensor = torch.randn(2048, 8192, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    # Reduce along second dimension (features dimension)
    dim = 1
    return [dim]