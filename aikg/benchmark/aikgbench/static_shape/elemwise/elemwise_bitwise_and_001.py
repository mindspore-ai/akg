import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise bitwise AND with broadcast (2D).
    Medium scale: e3
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 2D bitwise AND operation
        return torch.bitwise_and(input1, input2)


def get_inputs():
    # Medium scale: 64 * 64 ≈ e3

    input1 = torch.randint(0, 255, (64, 64), dtype=torch.int16)
    input2 = torch.randint(0, 255, (1, 1), dtype=torch.int16)
    return [input1, input2]


def get_init_inputs():
    return []

