import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise multiplication with middle dimension broadcast (3D).
    Medium scale: e7
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D multiplication operation, middle dimension broadcast
        return input1 * input2


def get_inputs():
    # Large scale: 128 * 256 * 256 ≈ e6

    input1 = torch.randn(128, 256, 256, dtype=torch.float16)
    input2 = torch.randn(128, 1, 256, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []


