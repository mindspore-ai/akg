import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.max(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the maximum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each maximum value found.
        # This operation is commonly used in neural networks for:
        # - Max pooling in convolutional networks
        # - Finding the most activated neuron in a layer
        # - Attention mechanisms in transformers
        return torch.max(var, self.dim)


def get_inputs_dyn_list():
    # Max reduction along dimension 1 variation cases with both aligned and non-aligned shapes (large sizes)

    # Case 1: Large tensor size 512x512 (aligned)
    inputs1 = torch.randn(512, 512, dtype=torch.float32)

    # Case 2: Very large tensor size 1023x1023 (non-aligned)
    inputs2 = torch.randn(1023, 1023, dtype=torch.float32)

    # Case 3: Very large tensor size 1024x1024 (aligned)
    inputs3 = torch.randn(1024, 1024, dtype=torch.float32)

    # Case 4: Extreme tensor size 4096x4096 (aligned)
    inputs4 = torch.randn(4096, 4096, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
    ]

def get_init_inputs():
    # Fixed parameters for max reduction along dimension 1
    dim = 1  # Reduce along second dimension (features dimension)
    return [dim]
