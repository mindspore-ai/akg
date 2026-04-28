import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise negation (1D, FP32).
    Small scale: e1
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 1D negation operation
        return torch.neg(input_tensor)


def get_inputs():
    # Small scale: 16 ≈ e1

    input_tensor = torch.randn(16, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    return []

