import torch
import torch.nn as nn


class Model(nn.Module):
    """
    C-style division with truncation (2D, FP32).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 2D cdiv operation, second dimension broadcast
        return torch.div(dividend, divisor, rounding_mode='trunc')


def get_inputs():
    # Medium scale: 1024 * 1024 ≈ e6

    dividend = torch.randn(1024, 1, dtype=torch.float32)
    divisor = torch.randn(1024, 1024, dtype=torch.float32) + 0.1
    return [dividend, divisor]


def get_init_inputs():
    return []

