import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var, var_scale=None, indices=None, updates=None, smooth_scales=None):
        # torch.min(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the minimum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each minimum value found.
        # This operation is commonly used in neural networks for:
        # - Finding the least activated neuron in a layer
        # - Implementing min-pooling operations
        # - Computing robust statistics in normalization layers
        return torch.min(var)


def get_inputs_dyn_list():
    # Global min reduction variation cases with both aligned and non-aligned shapes
    
    # Case 1: Large tensor size 256x256 (aligned)
    inputs1 = torch.randn(256, 256, dtype=torch.float32)

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
    # No parameters needed for global min
    return []