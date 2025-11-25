import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, eps=1e-5):
        super(Model, self).__init__()
        self.eps = eps

    def forward(self, x, residual, weight):
        # Fused operation: fused_add_rms_norm
        # Computes RMS normalization with addition in a single operation
        # This is more efficient than computing add and RMS norm separately
        # Fused add RMS normalization is commonly used in transformer models like LLaMA,
        # where the input and residual are added before applying RMS normalization.
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return weight * hidden_states


def get_inputs_dyn_list():
    # Fixed hidden dimension for all cases
    hidden_dim = 4096
    weight = torch.randn(hidden_dim, dtype=torch.float32)
    
    # Case 1: Small batch (non-aligned batch)
    shape1 = (15, hidden_dim)
    x1 = torch.randn(shape1, dtype=torch.float32)
    residual1 = torch.randn(shape1, dtype=torch.float32)
    
    # Case 2: Small batch (aligned batch)
    shape2 = (16, hidden_dim)
    x2 = torch.randn(shape2, dtype=torch.float32)
    residual2 = torch.randn(shape2, dtype=torch.float32)
    
    # Case 3: Medium batch (non-aligned batch)
    shape3 = (127, hidden_dim)
    x3 = torch.randn(shape3, dtype=torch.float32)
    residual3 = torch.randn(shape3, dtype=torch.float32)
    
    # Case 4: Large batch (aligned batch)
    shape4 = (512, hidden_dim)
    x4 = torch.randn(shape4, dtype=torch.float32)
    residual4 = torch.randn(shape4, dtype=torch.float32)
    
    # Case 5: Very large batch (non-aligned batch)
    shape5 = (1023, hidden_dim)
    x5 = torch.randn(shape5, dtype=torch.float32)
    residual5 = torch.randn(shape5, dtype=torch.float32)
    
    return [
        [x1, residual1, weight],
        [x2, residual2, weight],
        [x3, residual3, weight],
        [x4, residual4, weight],
        [x5, residual5, weight]
    ]


def get_init_inputs():
    # Only eps parameter needed for initialization
    eps = 1e-5
    return [eps]