import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor1, tensor2):
        # torch.cat(tensors, dim=0, *, out=None)
        # Concatenates the given sequence of tensors in the given dimension.
        # This operation is commonly used in neural networks for:
        # - Combining feature maps
        # - Merging batch dimensions
        # - Implementing skip connections
        return torch.cat([tensor1, tensor2], dim=-1)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 2048
    # Two tensors are concatenated along the feature dimension to get (1024, 4096)
    tensor1 = torch.randn(1024, 2048, dtype=torch.float32)
    tensor2 = torch.randn(1024, 2048, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    # No parameters needed for cat
    return []
