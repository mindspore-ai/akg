import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise negation (2D, FP32).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 2D negation operation
        return torch.neg(input_tensor)


def get_inputs():
    # Medium scale: 1024 * 4096 ≈ e6

    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for neg
    return []