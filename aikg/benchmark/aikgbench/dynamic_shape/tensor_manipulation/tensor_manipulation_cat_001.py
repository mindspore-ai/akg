import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input1, input2):
        # torch.cat(tensors, dim=1, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps from different sources
        # - Implementing skip connections in ResNet-like architectures
        # - Merging batch dimensions or sequence dimensions
        return torch.cat([input1, input2], dim=self.dim)


def get_inputs_dyn_list():
    # Two tensors are concatenated along the feature dimension (dim=1) to get (128, 512), (511, 1022), etc.
    # Small shape case - concatenate along feature dimension
    inputs1_1 = torch.randn(128, 256, dtype=torch.float32)
    inputs1_2 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case - concatenate along feature dimension
    inputs2_1 = torch.randn(511, 511, dtype=torch.float32)
    inputs2_2 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case - concatenate along feature dimension
    inputs3_1 = torch.randn(512, 2048, dtype=torch.float32)
    inputs3_2 = torch.randn(512, 2048, dtype=torch.float32)
    # Standard Large shape case - concatenate along feature dimension
    inputs4_1 = torch.randn(1024, 2048, dtype=torch.float32)
    inputs4_2 = torch.randn(1024, 2048, dtype=torch.float32)
    # Large shape case - concatenate along feature dimension
    inputs5_1 = torch.randn(2048, 4096, dtype=torch.float32)
    inputs5_2 = torch.randn(2048, 4096, dtype=torch.float32)

    return [
        [inputs1_1, inputs1_2],
        [inputs2_1, inputs2_2],
        [inputs3_1, inputs3_2],
        [inputs4_1, inputs4_2],
        [inputs5_1, inputs5_2]
    ]


def get_init_inputs():
    # Specific dim value for concatenation
    dim = 1  # Concatenate along the feature dimension
    return [dim]