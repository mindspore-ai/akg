import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, var_scale=None, indices=None, updates=None, smooth_scales=None):
        # torch.argmin(input, dim, keepdim=False)
        # Returns the indices of the minimum values of all elements in the input tensor
        # or along a dimension if specified.
        # This operation is commonly used in neural networks for:
        # - Finding the least probable token in sequence generation
        # - Implementing specialized selection mechanisms
        # - Computing robust statistics in normalization layers
        return torch.argmin(var, dim=self.dim)


def get_inputs_dyn_list():
    # Argmin reduction along dimension 0 variation cases with both aligned and non-aligned shapes
    
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
    # Fixed parameters for argmin reduction along dimension 0
    dim = 0  # Reduce along first dimension (batch dimension)
    return [dim]