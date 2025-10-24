import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Element-wise modulus operation (2D, FP32).
    Medium scale: e6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, divisor):
        # 2D modulus operation
        return torch.fmod(input_tensor, divisor)


def get_inputs():
    # Medium scale: 1024 * 4096 ≈ e6

    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    divisor = torch.randn(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to avoid division by zero
    return [input_tensor, divisor]


def get_init_inputs():
    # No parameters needed for mod
    return []