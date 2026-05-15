import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, var_scale=None, indices=None, updates=None, smooth_scales=None):
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
    # Max reduction along dimension 1 variation cases with both aligned and non-aligned shapes (small sizes)
    
    # Case 1: Very small tensor size 8x8 (aligned)
    inputs1 = torch.randn(8, 8, dtype=torch.float32)
    
    # Case 2: Small tensor size 15x15 (non-aligned)
    inputs2 = torch.randn(15, 15, dtype=torch.float32)

    # Case 3: Small tensor size 32x32 (aligned)
    inputs3 = torch.randn(32, 32, dtype=torch.float32)
    
    # Case 4: Medium tensor size 128x128 (aligned)
    inputs4 = torch.randn(128, 128, dtype=torch.float32)
    
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