import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y):
        # Fused operation: silu_and_mul
        # Computes silu(x) * y in a single operation
        # This is more efficient than computing silu and mul separately
        # SiLU (Sigmoid Linear Unit) is an activation function commonly used in neural networks.
        # The fused operation combines the activation with element-wise multiplication,
        # which is often used in feed-forward networks of transformers.
        silu_x = torch.nn.functional.silu(x)
        return torch.mul(silu_x, y)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    x = torch.randn(1024, 4096, dtype=torch.float32)
    y = torch.randn(1024, 4096, dtype=torch.float32)
    return [x, y]


def get_init_inputs():
    # No parameters needed for silu_and_mul
    return []