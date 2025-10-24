import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise absolute value (3D, FP16).
    Medium scale: e5
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D absolute value operation
        return torch.abs(input_tensor)


def get_inputs():
    # Medium scale: 16 * 16 * 1024 ≈ e5

    input_tensor = torch.randn(16, 16, 1024, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for abs
    return []

