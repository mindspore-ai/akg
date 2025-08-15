import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.prod(input, dim, keepdim=False, dtype=None)
        # Returns the product of each row of the input tensor in the given dimension dim.
        # This operation is commonly used in neural networks for:
        # - Computing joint probabilities
        # - Implementing specialized aggregation functions
        # - Mathematical transformations in certain layers
        return torch.prod(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 128, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    dim = 1  # Reduce along second dimension (features dimension)
    return [dim]