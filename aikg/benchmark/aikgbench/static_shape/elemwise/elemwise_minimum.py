import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # torch.minimum(input, other, *, out=None)
        # Computes the element-wise minimum of input and other.
        # This operation is commonly used in neural networks for:
        # - Clamping values to a maximum threshold
        # - Implementing certain activation functions
        # - Combining feature maps with element-wise minimum
        return torch.minimum(input1, input2)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input1 = torch.randn(1024, 4096, dtype=torch.float32)
    input2 = torch.randn(1024, 4096, dtype=torch.float32)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for minimum
    return []