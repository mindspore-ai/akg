import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input1, input2, input3):
        # torch.cat(tensors, dim=0, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps from different sources
        # - Implementing skip connections in ResNet-like architectures
        # - Merging batch dimensions or sequence dimensions
        return torch.cat([input1, input2, input3], dim=self.dim)


def get_inputs_dyn_list():
    # Two tensors are concatenated along the batch dimension (dim=0) to get (256, 256), (1022, 511), etc.
    # Small shape case - concatenate along batch dimension
    inputs1_1 = torch.randn(128, 256, dtype=torch.float32)
    inputs1_2 = torch.randn(128, 256, dtype=torch.float32)
    inputs1_3 = torch.randn(128, 256, dtype=torch.float32)
    # Non-aligned shape case - concatenate along batch dimension
    inputs2_1 = torch.randn(511, 511, dtype=torch.float32)
    inputs2_2 = torch.randn(511, 511, dtype=torch.float32)
    inputs2_3 = torch.randn(511, 511, dtype=torch.float32)
    # Middle shape case - concatenate along batch dimension
    inputs3_1 = torch.randn(512, 4096, dtype=torch.float32)
    inputs3_2 = torch.randn(512, 4096, dtype=torch.float32)
    inputs3_3 = torch.randn(512, 4096, dtype=torch.float32)
    # Standard Large shape case - concatenate along batch dimension
    inputs4_1 = torch.randn(1024, 4096, dtype=torch.float32)
    inputs4_2 = torch.randn(1024, 4096, dtype=torch.float32)
    inputs4_3 = torch.randn(1024, 4096, dtype=torch.float32)
    # Large shape case - concatenate along batch dimension
    inputs5_1 = torch.randn(2048, 8192, dtype=torch.float32)
    inputs5_2 = torch.randn(2048, 8192, dtype=torch.float32)
    inputs5_3 = torch.randn(2048, 8192, dtype=torch.float32)

    return [
        [inputs1_1, inputs1_2, inputs1_3],
        [inputs2_1, inputs2_2, inputs2_3],
        [inputs3_1, inputs3_2, inputs3_3],
        [inputs4_1, inputs4_2, inputs4_3],
        [inputs5_1, inputs5_2, inputs5_3]
    ]


def get_init_inputs():
    # Specific dim value for concatenation
    dim = 0  # Concatenate along the batch dimension
    return [dim]