import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor):
        # torch.argmin(input, dim, keepdim=False)
        # Returns the indices of the minimum values of all elements in the input tensor
        # or along a dimension if specified.
        # This operation is commonly used in neural networks for:
        # - Finding the least probable token in sequence generation
        # - Implementing specialized selection mechanisms
        # - Computing robust statistics in normalization layers
        return torch.argmin(input_tensor, dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Specific dim value for reduction
    # Reduce along second dimension (features dimension)
    dim = 1
    return [dim]