import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise clamp operation (1D, FP16).
    Small scale: e1
    """
    def __init__(self, min_val=-1.0, max_val=1.0):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input_tensor):
        # 1D clamp operation
        return torch.clamp(input_tensor, self.min_val, self.max_val)


def get_inputs():
    # Small scale: 16 ≈ e1

    input_tensor = torch.randn(16, dtype=torch.float16)
    return [input_tensor]


def get_init_inputs():
    return [-1.0, 1.0]

