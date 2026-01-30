import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, var_scale=None, indices=None, updates=None, smooth_scales=None):
        # torch.mean(input, dim, keepdim=False, dtype=None)
        # Returns the mean value of all elements in the input tensor or along the specified dimension.
        # This operation is commonly used in neural networks for:
        # - Computing loss functions (e.g., mean squared error)
        # - Normalizing activations across batch dimensions
        # - Pooling operations in convolutional networks
        return torch.mean(var, dim=self.dim)


def get_inputs_dyn_list():
    # Mean reduction variation cases with both aligned and non-aligned shapes (all dimensions)
    
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
    # Fixed parameters for mean reduction
    dim = None  # Reduce all dimensions
    return [dim]