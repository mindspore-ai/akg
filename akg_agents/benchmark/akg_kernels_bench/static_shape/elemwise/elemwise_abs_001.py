import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise absolute value (2D, int32).
    Large scale: e7
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 2D absolute value operation
        return torch.abs(input_tensor)


def get_inputs():
    # Medium scale: 4096 * 4096 ≈ e7

    input_tensor = torch.randint(-128, 127, (4096, 4096), dtype=torch.int32)
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for abs
    return []