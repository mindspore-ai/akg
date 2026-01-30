import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, tensor1, tensor2, tensor3):
        # torch.cat(tensors, dim=0, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps from different sources
        # - Implementing skip connections in ResNet-like architectures
        # - Merging batch dimensions or sequence dimensions
        return torch.cat([tensor1, tensor2, tensor3], dim=self.dim)


def get_inputs():
    # Batch size: 512
    # Hidden dimension: 4096
    # Three tensors are concatenated along the batch dimension to get (1536, 4096)
    tensor1 = torch.randn(512, 4096, dtype=torch.float32)
    tensor2 = torch.randn(512, 4096, dtype=torch.float32)
    tensor3 = torch.randn(512, 4096, dtype=torch.float32)
    return [tensor1, tensor2, tensor3]


def get_init_inputs():
    # Specific dim value for concatenation
    dim = 0  # Concatenate along the batch dimension
    return [dim]