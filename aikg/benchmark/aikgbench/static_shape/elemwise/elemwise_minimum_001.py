import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise minimum (3D, int64).
    Medium scale: e5
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D minimum operation, first dimension broadcast
        return torch.minimum(input1, input2)


def get_inputs():
    # Medium scale: 64 * 128 * 64 ≈ e5

    input1 = torch.randint(-128, 127, (1, 128, 64), dtype=torch.int64)
    input2 = torch.randint(-128, 127, (64, 128, 64), dtype=torch.int64)
    return [input1, input2]


def get_init_inputs():
    return []

