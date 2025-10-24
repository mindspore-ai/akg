import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise division (2D, FP16).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 2D division with FP16
        return dividend / divisor


def get_inputs():
    # Medium scale: 1024 * 1024 ≈ e6

    dividend = torch.randn(1024, 1024, dtype=torch.float16)
    divisor = torch.randn(1024, 1024, dtype=torch.float16) + 1.0
    return [dividend, divisor]


def get_init_inputs():
    return []

