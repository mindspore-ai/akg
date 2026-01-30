import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, layer_shape, weight, eps=1e-5):
        super(Model, self).__init__()
        self.layer_shape = layer_shape
        self.weight = weight
        self.eps = eps

    def forward(self, x, residual):
        # Fused operation: fused_add_rms_norm
        # Computes RMS normalization with addition in a single operation
        # This is more efficient than computing add and RMS norm separately
        # Fused add RMS normalization is commonly used in transformer models like LLaMA,
        # where the input and residual are added before applying RMS normalization.
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    shape = (1024, 4096)
    x = torch.randn(shape, dtype=torch.float32)
    residual = torch.randn(shape, dtype=torch.float32)
    return [x, residual]


def get_init_inputs():
    # Parameters for fused_add_rms_norm
    shape = (1024, 4096)
    layer_shape = (shape[-1],)  # Normalizing over the last dimension
    weight = torch.randn(layer_shape, dtype=torch.float32)
    eps = 1e-5  # Small epsilon value for numerical stability
    
    return [layer_shape, weight, eps]