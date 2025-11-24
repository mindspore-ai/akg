import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var, cmp_val):
        # torch.sum(input, dim, keepdim=False, dtype=None)
        # Returns the sum of each row of the input tensor in the given dimension dim.
        # This operation is commonly used in neural networks for:
        # - Counting elements that meet certain conditions
        # - Computing statistics on boolean masks
        # - Implementing custom reduction operations
        condition = (var == cmp_val)
        return torch.sum(condition, dim=self.dim)


def get_inputs_dyn_list():
    # Count vector along dimension 1 variation cases with both aligned and non-aligned shapes

    # Case 1: Very large tensor size 1023x1023 (non-aligned)
    inputs1 = torch.randn(1023, 1023, dtype=torch.float32)
    cmp_val1 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare against

    # Case 2: Very large tensor size 1024x1024 (aligned)
    inputs2 = torch.randn(1024, 1024, dtype=torch.float32)
    cmp_val2 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare against

    # Case 3: Extreme tensor size 4096x4096 (aligned)
    inputs3 = torch.randn(4096, 4096, dtype=torch.float32)
    cmp_val3 = torch.tensor(0.5, dtype=torch.float32)  # Value to compare against

    return [
        [inputs1, cmp_val1],
        [inputs2, cmp_val2],
        [inputs3, cmp_val3],
    ]


def get_init_inputs():
    # Fixed parameters for count vector along dimension 1
    dim = 1  # Reduce along second dimension (features dimension)
    return [dim]
