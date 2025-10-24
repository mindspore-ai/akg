import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Floor division (3D, int64).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dividend, divisor):
        # 3D floor division, last two dimensions broadcast
        return torch.floor_divide(dividend, divisor)


def get_inputs():
    # Medium scale: 128 * 128 * 64 ≈ e6

    dividend = torch.randn(128, 1, 1, dtype=torch.int64)
    divisor = torch.randn(128, 128, 64, dtype=torch.int64) + 0.1
    return [dividend.to(torch.int64), divisor.to(torch.int64)]


def get_init_inputs():
    return []

