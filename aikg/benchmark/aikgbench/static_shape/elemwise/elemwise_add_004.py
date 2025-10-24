import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise addition with front broadcasting (1D).
    Small scale: e2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # input1: (10000,), input2: (1,) -> broadcast from front
        return input1 + input2


def get_inputs():
    # Small scale: 128 ≈ e2

    input1 = torch.randn(128, dtype=torch.float16)
    input2 = torch.randn(1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    # No parameters needed for add
    return []


