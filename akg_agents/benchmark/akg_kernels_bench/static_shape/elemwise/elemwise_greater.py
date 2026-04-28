import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.gt(input, other, *, out=None)
        # Computes input > other elementwise.
        # This operation is commonly used in neural networks for:
        # - Implementing comparison operations
        # - Creating masks for conditional operations
        # - Thresholding operations
        return torch.gt(input1, input2)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randn(1024, 4096, dtype=torch.float32)
    input2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for greater
    return []