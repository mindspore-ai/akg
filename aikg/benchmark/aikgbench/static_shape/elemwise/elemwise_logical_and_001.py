import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise logical AND with broadcast (2D).
    Medium scale: e5
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # input1: (512, 512), input2: (1, 512) -> horizontal broadcast
        return torch.logical_and(input1, input2)


def get_inputs():
    # Medium scale: 512 * 512 ≈ e5

    input1 = torch.randint(0, 2, (512, 512), dtype=torch.bool)
    input2 = torch.randint(0, 2, (1, 512), dtype=torch.bool)
    return [input1, input2]


def get_init_inputs():
    return []

