import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise sine (1D, FP32).
    Small scale: e2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 1D sine operation
        return torch.sin(input_tensor)


def get_inputs():
    # Small scale: 256 ≈ e2

    input_tensor = torch.randn(256, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    return []

