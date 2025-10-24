import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise less than comparison (1D, int8).
    Medium scale: e4
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 1D less than comparison
        return torch.lt(input1, input2)


def get_inputs():
    # Medium scale: 16384 ≈ e4

    input1 = torch.randint(-128, 127, (16384), dtype=torch.int8)
    input2 = torch.randint(-128, 127, (16384), dtype=torch.int8)
    return [input1, input2]


def get_init_inputs():
    return []

