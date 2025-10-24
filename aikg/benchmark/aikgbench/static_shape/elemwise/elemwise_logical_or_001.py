import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise logical OR with broadcast (1D).
    Small scale: e2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 1D logical OR operation
        return torch.logical_or(input1, input2)


def get_inputs():
    # Small scale: 128 ≈ e2

    input1 = torch.randint(0, 2, (128,), dtype=torch.bool)
    input2 = torch.randint(0, 2, (128,), dtype=torch.bool)
    return [input1, input2]


def get_init_inputs():
    return []

