import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.max(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the maximum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each maximum value found.
        # This is a max operation with no specific dimension (returns global maximum).
        # Max operations are commonly used in neural networks for:
        # - Max pooling in convolutional networks
        # - Finding the most activated neuron in a layer
        # - Attention mechanisms in transformers
        return torch.max(input_tensor)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for global max
    return []