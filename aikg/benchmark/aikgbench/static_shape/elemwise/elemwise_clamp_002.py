import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise clamp operation (3D, FP16).
    Medium scale: e3
    """
    def __init__(self, min_val=-2.0, max_val=2.0):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input_tensor):
        # 3D clamp operation
        return torch.clamp(input_tensor, self.min_val, self.max_val)


def get_inputs():
    # Medium scale: 16 * 16 * 16 ≈ e3

    input_tensor = torch.randn(16, 16, 16, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return [-2.0, 2.0]

