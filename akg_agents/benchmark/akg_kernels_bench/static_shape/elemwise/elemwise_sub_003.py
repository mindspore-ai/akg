import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise subtraction (2D, FP16).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 2D subtraction operation
        return input1 - input2


def get_inputs():
    input1 = torch.randn(2048, 131072, dtype=torch.float16)
    input2 = torch.randn(2048, 1, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []

