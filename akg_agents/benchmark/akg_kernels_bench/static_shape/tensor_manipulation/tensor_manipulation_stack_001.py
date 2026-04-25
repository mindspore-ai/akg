import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=0):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, tensor1, tensor2):
        # torch.stack(tensors, dim=0, *, out=None)
        # Concatenates a sequence of tensors along a new dimension.
        # This is a stack operation along dimension 0 (new batch dimension).
        # Stack operations are commonly used in neural networks for:
        # - Combining multiple feature maps into a new dimension
        # - Creating batch dimensions
        # - Implementing certain attention mechanisms
        return torch.stack([tensor1, tensor2], dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Two tensors are stacked along a new dimension to get (2, 1024, 4096)
    tensor1 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    # Parameters needed for stack
    dim = 0  # Stack dimension
    return [dim]