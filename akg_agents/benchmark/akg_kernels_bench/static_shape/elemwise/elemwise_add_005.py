import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise addition with front broadcasting (3D).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # input1: (32, 128, 256), input2: (1, 128, 256) -> front broadcast
        return input1 + input2


def get_inputs():
    # Medium scale: 32 * 128 * 256 ≈ e6

    input1 = torch.randn(32, 128, 256, dtype=torch.float16)
    input2 = torch.randn(1, 128, 256, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for add
    return []

