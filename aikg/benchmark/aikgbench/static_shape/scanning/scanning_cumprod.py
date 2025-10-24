import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.cumprod(input, dim, *, dtype=None, out=None)
        # Returns the cumulative product of elements of input in the dimension dim.
        # The cumulative product for each element is the product of all previous elements
        # in the specified dimension up to and including the current element.
        # This operation is commonly used in neural networks for:
        # - Computing sequential products in probabilistic models
        # - Implementing certain mathematical transformations
        # - Computing joint probabilities in Bayesian networks
        return torch.cumprod(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for cumulative product
    # Compute cumulative product along second dimension (features dimension)
    dim = 1
    return [dim]