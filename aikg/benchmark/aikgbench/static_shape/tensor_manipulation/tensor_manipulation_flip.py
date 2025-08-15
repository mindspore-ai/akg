import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dims=[-1]):
        super(Model, self).__init__()
        self.dims = dims

    def forward(self, input_tensor):
        # torch.flip(input, dims)
        # Reverse the order of a n-D tensor along given axis in dims.
        # This operation is commonly used in neural networks for:
        # - Reversing sequences in RNNs
        # - Data augmentation techniques
        # - Implementing certain convolution operations
        return torch.flip(input_tensor, dims=self.dims)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # Parameters needed for flip
    dims = [-1]  # Flip dimensions
    return [dims]