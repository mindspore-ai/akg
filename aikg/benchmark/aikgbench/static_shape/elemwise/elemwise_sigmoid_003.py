import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Sigmoid activation (1D, FP16).
    Small scale: e2
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 1D sigmoid activation
        return torch.sigmoid(input_tensor)


def get_inputs():
    # Small scale: 128 ≈ e2

    input_tensor = torch.randn(128, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return []

