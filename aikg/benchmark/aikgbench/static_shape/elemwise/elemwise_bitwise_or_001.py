import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise bitwise OR (3D).
    Medium scale: 16 * 16 * 16 ≈ e4
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D bitwise OR operation
        return torch.bitwise_or(input1, input2)


def get_inputs():
    # Medium scale: 16 * 16 * 16 ≈ e4
    input1 = torch.randint(0, 255, (1, 16, 1), dtype=torch.bool)
    input2 = torch.randint(0, 255, (16, 16, 16), dtype=torch.bool)
    return [input1, input2]


def get_init_inputs():
    return []

