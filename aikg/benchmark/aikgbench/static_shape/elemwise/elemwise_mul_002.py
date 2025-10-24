import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise multiplication with vertical broadcast (3D).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D multiplication operation, vertical broadcast
        return input1 * input2


def get_inputs():
    # Medium scale: 64 * 128 * 128 ≈ e6

    input1 = torch.randn(64, 128, 128, dtype=torch.float16)
    input2 = torch.randn(64, 1, 1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []


