import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, cmp_val):
        # torch.eq(input, other, *, out=None)
        # Computes element-wise equality.
        # Then we count the number of True values along the specified dimension.
        # This operation is commonly used in neural networks for:
        # - Counting elements that meet specific criteria
        # - Computing statistics on boolean masks
        # - Implementing certain counting mechanisms
        condition = (input_tensor == cmp_val)
        return torch.sum(condition, dim=self.dim)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    cmp_val = torch.tensor(0.5, dtype=torch.float32)  # Value to compare with
    return [input_tensor, cmp_val]


def get_init_inputs():
    # Specific dim value for reduction
    dim = 0  # Count along first dimension (batch dimension)
    return [dim]