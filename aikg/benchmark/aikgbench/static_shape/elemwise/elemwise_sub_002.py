import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise subtraction (3D, bfloat16).
    Large scale: e8
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 3D subtraction operation, broadcasting on front and middle dimensions
        return input1 - input2


def get_inputs():
    # Large scale: 256 * 1024 * 1024 ≈ e8

    input1 = torch.randn(256, 1024, 1024, dtype=torch.bfloat16)
    input2 = torch.randn(1, 1, 1024, dtype=torch.bfloat16)
    return [input1, input2]


def get_init_inputs():
    return []


