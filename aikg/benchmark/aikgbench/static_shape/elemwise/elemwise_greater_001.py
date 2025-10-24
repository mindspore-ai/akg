import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise greater than comparison (3D, bool).
    Medium scale: e5
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D greater than comparison, no broadcast
        return torch.gt(input1, input2)


def get_inputs():
    # Medium scale: 64 * 64 * 64 ≈ e5

    input1 = torch.randint(0, 2, (64, 64, 64), dtype=torch.bool)
    input2 = torch.randint(0, 2, (64, 64, 64), dtype=torch.bool)
    return [input1, input2]


def get_init_inputs():
    return []

