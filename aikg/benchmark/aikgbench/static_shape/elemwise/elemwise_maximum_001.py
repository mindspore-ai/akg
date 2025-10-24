import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise maximum (2D, FP16).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1, input2):
        # 2D maximum operation
        return torch.maximum(input1, input2)


def get_inputs():
    # Medium scale: 2048 * 2048 ≈ e6

    input1 = torch.randn(2048, 2048, dtype=torch.float16)
    input2 = torch.randn(1, 2048, dtype=torch.float16)
    return [input1, input2]


def get_init_inputs():
    return []

