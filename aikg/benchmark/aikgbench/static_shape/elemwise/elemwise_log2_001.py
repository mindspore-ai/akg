import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise logarithm to base 2 (3D, bfloat16).
    Large scale: e7
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # 3D logarithm to base 2 operation
        return torch.log2(input_tensor)


def get_inputs():
    # Large scale: 4096 * 4096 ≈ e7

    input_tensor = torch.rand(4096, 4096, dtype=torch.bfloat16) + 1e-6  # Adding small value to ensure positive values
    return [input_tensor]


def get_init_inputs():
    # No parameters needed for log2
    return []