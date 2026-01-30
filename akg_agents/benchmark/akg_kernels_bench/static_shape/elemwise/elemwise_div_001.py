import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise division with broadcast (2D, FP16).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 2D division operation
        return dividend / divisor


def get_inputs():

    dividend = torch.randn(131072, 16, dtype=torch.float16)
    divisor = torch.randn(1, 16, dtype=torch.float16) + 1.0
    return [dividend, divisor]


def get_init_inputs():
    return []

