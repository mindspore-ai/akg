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


def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    x1 = torch.randn(15, 1344, dtype=torch.float32)
    y1 = torch.randn(15, 1344, dtype=torch.float32)
    
    # Case 2: Small batch, large hidden (aligned batch)
    x2 = torch.randn(16, 4096, dtype=torch.float32)
    y2 = torch.randn(16, 4096, dtype=torch.float32)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    x3 = torch.randn(127, 2688, dtype=torch.float32)
    y3 = torch.randn(127, 2688, dtype=torch.float32)
    
    # Case 4: Large batch, large hidden (aligned batch)
    x4 = torch.randn(512, 5120, dtype=torch.float32)
    y4 = torch.randn(512, 5120, dtype=torch.float32)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    x5 = torch.randn(1023, 8192, dtype=torch.float32)
    y5 = torch.randn(1023, 8192, dtype=torch.float32)
    
    return [
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4],
        [x5, y5]
    ]


def get_init_inputs():
    # No parameters needed for silu_and_mul
    return []