import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.bitwise_and(input, other, *, out=None)
        # Computes the element-wise bitwise AND of the given input tensors.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Implementing bit manipulation operations
        # - Creating bit masks
        # - Low-level data processing
        return torch.bitwise_and(input1, input2)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randint(0, 256, (1024, 4096), dtype=torch.int32)
    input2 = torch.randint(0, 256, (1024, 4096), dtype=torch.int32)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for bitwise_and
    return []