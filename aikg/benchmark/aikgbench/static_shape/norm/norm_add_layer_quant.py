import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + Layer Normalization + Quantization fused operation.
    Add residual, layer-norm, then quantize.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma, beta, scale, zero_point):
        # Add residual connection
        x_added = x + residual

        # Layer normalization and affine
        mean = x_added.mean(dim=-1, keepdim=True)
        var = x_added.var(dim=-1, keepdim=True, unbiased=False)
        output = ((x_added - mean) / torch.sqrt(var + self.epsilon)) * gamma + beta

        # Quantize to int8
        output_quantized = torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
        return output_quantized


def get_inputs():
    # Batch size: 32, Sequence length: 1024, Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)
    scale = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point = torch.randn(1, dtype=torch.float32) * 10
    return [x, residual, gamma, beta, scale, zero_point]


def get_init_inputs():
    return [1e-6]  # epsilon