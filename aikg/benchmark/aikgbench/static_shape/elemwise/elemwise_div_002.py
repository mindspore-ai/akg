import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise division (3D, FP16) with full broadcast (scalar-like tensor).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 3D division operation, full broadcast
        return dividend / divisor


def get_inputs():
    # Medium scale: 64 * 128 * 256 ≈ e6

    dividend = torch.randn(64, 128, 256, dtype=torch.float16)
    divisor = torch.randn(1, 1, 1, dtype=torch.float16) + 1.0
    return [dividend, divisor]


def get_init_inputs():
    return []


