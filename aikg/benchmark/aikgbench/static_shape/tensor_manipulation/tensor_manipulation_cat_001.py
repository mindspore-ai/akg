import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, tensor1, tensor2):
        # torch.cat(tensors, dim=0, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps from different sources
        # - Implementing skip connections in ResNet-like architectures
        # - Merging batch dimensions or sequence dimensions
        return torch.cat([tensor1, tensor2], dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 2048
    # Two tensors are concatenated along the feature dimension to get (1024, 4096)
    tensor1 = torch.randn(1024, 2048, dtype=torch.float32)
    tensor2 = torch.randn(1024, 2048, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    # Specific dim value for concatenation
    dim = 1  # Concatenate along the feature dimension
    return [dim]