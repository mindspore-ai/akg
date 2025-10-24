import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise ceil operation (1D, bfloat16).
    Small scale: e1
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 1D ceil operation
        return torch.ceil(input_tensor)


def get_inputs():
    # Small scale: 16 ≈ e1

    input_tensor = torch.randn(16, dtype=torch.bfloat16)
    return [input_tensor]


def get_init_inputs():
    return []

