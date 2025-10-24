import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise ceil operation (3D, FP16).
    Large scale: e7
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D ceil operation
        return torch.ceil(input_tensor)


def get_inputs():
    # Large scale: 256 * 1024 * 1024 ≈ e8

    input_tensor = torch.randn(256, 1024, 1024, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return []

