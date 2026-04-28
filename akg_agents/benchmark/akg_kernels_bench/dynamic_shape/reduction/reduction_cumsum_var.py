import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, var_scale=None, indices=None, updates=None, smooth_scales=None):
        # torch.cumsum(input, dim, *, dtype=None, out=None)
        # Returns the cumulative sum of elements of input in the dimension dim.
        # The cumulative sum for each element is the sum of all previous elements
        # in the specified dimension up to and including the current element.
        # This operation is commonly used in neural networks for:
        # - Computing prefix sums in attention mechanisms
        # - Implementing certain mathematical transformations
        # - Computing running totals in recurrent networks
        return torch.cumsum(var, dim=self.dim)


def get_inputs_dyn_list():
    # Cumulative sum along dimension 1 variation cases with both aligned and non-aligned shapes
    
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
    # Fixed parameters for cumulative sum along dimension 1
    dim = 1  # Compute cumulative sum along second dimension (features dimension)
    return [dim]