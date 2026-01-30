import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, dim: int):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.amax(input, dim, keepdim=False)
        # Returns the maximum value of each row of the input tensor in the given dimension(s) dim.
        # This function is very similar to torch.max(), but differs in how gradients are propagated.
        # It is commonly used in neural networks for:
        # - Robust pooling operations
        # - Finding extreme values in feature maps
        return torch.amax(var, self.dim)


def get_inputs_dyn_list():
    # Amax reduction variation cases with both aligned and non-aligned shapes

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
    # Fixed parameters for amax reduction
    dim = 1  # Reduce along second dimension (features dimension)
    return [dim]
