import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, layer_shape):
        super(Model, self).__init__()
        self.layer_shape = layer_shape

    def forward(self, inp, residual, weight, bias):
        # Fused operation: add_layer_norm
        # Computes layer_norm(inp + residual, layer_shape, weight, bias) in a single operation
        # More efficient than computing add and layer_norm separately
        added = inp + residual
        return torch.layer_norm(added, self.layer_shape, weight, bias)


def get_inputs_dyn_list():
    # Fixed hidden dimension for all cases
    hidden_dim = 4096
    weight = torch.randn(hidden_dim, dtype=torch.float32)
    bias = torch.randn(hidden_dim, dtype=torch.float32)
    
    # Case 1: Small batch (non-aligned batch)
    shape1 = (15, hidden_dim)
    inp1 = torch.randn(shape1, dtype=torch.float32)
    residual1 = torch.randn(shape1, dtype=torch.float32)
    
    # Case 2: Small batch (aligned batch)
    shape2 = (16, hidden_dim)
    inp2 = torch.randn(shape2, dtype=torch.float32)
    residual2 = torch.randn(shape2, dtype=torch.float32)
    
    # Case 3: Medium batch (non-aligned batch)
    shape3 = (127, hidden_dim)
    inp3 = torch.randn(shape3, dtype=torch.float32)
    residual3 = torch.randn(shape3, dtype=torch.float32)
    
    # Case 4: Large batch (aligned batch)
    shape4 = (512, hidden_dim)
    inp4 = torch.randn(shape4, dtype=torch.float32)
    residual4 = torch.randn(shape4, dtype=torch.float32)
    
    # Case 5: Very large batch (non-aligned batch)
    shape5 = (1023, hidden_dim)
    inp5 = torch.randn(shape5, dtype=torch.float32)
    residual5 = torch.randn(shape5, dtype=torch.float32)
    
    return [
        [inp1, residual1, weight, bias],
        [inp2, residual2, weight, bias],
        [inp3, residual3, weight, bias],
        [inp4, residual4, weight, bias],
        [inp5, residual5, weight, bias]
    ]


def get_init_inputs():
    # Fixed layer shape for all cases
    return [(4096,)]