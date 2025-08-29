import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.cumprod(input, dim, *, dtype=None, out=None)
        # Returns the cumulative product of elements of input in the dimension dim.
        # The cumulative product for each element is the product of all previous elements
        # in the specified dimension up to and including the current element.
        # This operation is commonly used in neural networks for:
        # - Computing sequential products in probabilistic models
        # - Implementing certain mathematical transformations
        # - Computing joint probabilities in Bayesian networks
        return torch.cumprod(var, dim=self.dim)


def get_inputs_dyn_list():
    # Cumulative product along dimension 1 variation cases with both aligned and non-aligned shapes

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
    # Fixed parameters for cumulative product along dimension 1
    dim = 1  # Compute cumulative product along second dimension (features dimension)
    return [dim]
