import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise natural logarithm (3D, FP16).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D log operation
        return torch.log(input_tensor)


def get_inputs():
    # Medium scale: 128 * 128 * 64 ≈ e6

    input_tensor = torch.rand(128, 128, 64, dtype=torch.float16) + 0.1
    return [input_tensor]


def get_init_inputs():
    return []

