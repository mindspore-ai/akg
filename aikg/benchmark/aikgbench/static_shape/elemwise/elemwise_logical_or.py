import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.logical_or(input, other, *, out=None)
        # Computes the element-wise logical OR of the given input tensors.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Combining multiple boolean masks
        # - Implementing complex conditional logic
        # - Creating union of conditions
        return torch.logical_or(input1, input2)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)
    input2 = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for logical_or
    return []