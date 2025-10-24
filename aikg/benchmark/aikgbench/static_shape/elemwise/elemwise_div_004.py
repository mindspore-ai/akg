import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise division (1D, FP16).
    Medium scale: e3
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 1D division with FP16
        return dividend / divisor


def get_inputs():
    # Medium scale: 1024 ≈ e3

    dividend = torch.randn(1024, dtype=torch.float16)
    divisor = torch.randn(1024, dtype=torch.float16) + 1.0
    return [dividend, divisor]


def get_init_inputs():
    return []

