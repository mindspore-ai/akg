import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise sine (3D, FP16).
    Large scale: e8
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D sine operation
        return torch.sin(input_tensor)


def get_inputs():
    # Large scale: 512 * 512 * 256 ≈ e7

    input_tensor = torch.randn(512, 512, 256, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return []

