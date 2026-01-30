import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.sum(input, dim, keepdim=False, dtype=None)
        # Returns the sum of each row of the input tensor in the given dimension dim.
        # This is a reduction operation along dimension 0.
        # Reduction operations are commonly used in neural networks for:
        # - Computing loss functions (e.g., mean squared error)
        # - Normalizing activations across batch dimensions
        # - Pooling operations in convolutional networks
        return torch.sum(var, self.dim)


def get_inputs_dyn_list():
    # Sum reduction variation cases with both aligned and non-aligned shapes (dim=0)

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
    # Fixed parameters for sum reduction
    dim = 0  # Reduce along first dimension (batch dimension)
    return [dim]
