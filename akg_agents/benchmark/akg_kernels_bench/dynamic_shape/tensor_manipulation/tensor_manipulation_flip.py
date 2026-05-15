import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dims=[-1]):
        super(Model, self).__init__()
        self.dims = dims

    def forward(self, inputs):
        # torch.flip(input, dims)
        # Reverse the order of a n-D tensor along given axis in dims.
        # This operation is commonly used in neural networks for:
        # - Reversing sequences in RNNs
        # - Data augmentation techniques
        # - Implementing certain convolution operations
        return torch.flip(inputs, dims=self.dims)


def get_inputs_dyn_list():
    # Three tensors are concatenated along the batch dimension to get (1536, 4096)
    # Small shape case
    inputs1 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case
    inputs2 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case
    inputs3 = torch.randn(512, 4096, dtype=torch.float32)
    # Standard Large shape case
    inputs4 = torch.randn(1024, 4096, dtype=torch.float32)
    # Large shape case
    inputs5 = torch.randn(2048, 8192, dtype=torch.float32)

    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5]
    ]


def get_init_inputs():
    # Parameters needed for flip
    dims = [-1]  # Flip dimensions
    return [dims]