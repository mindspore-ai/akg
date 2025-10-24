import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise reciprocal square root (3D, FP16).
    Large scale: e8
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D reciprocal square root operation
        return torch.rsqrt(input_tensor)


def get_inputs():
    # Large scale: 4096 * 2048 * 16 ≈ e8

    input_tensor = torch.randn(4096, 2048, 16, dtype=torch.float16).abs() + 0.1
    return [input_tensor]


def get_init_inputs():
    return []

