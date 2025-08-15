import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Quantized Add + Layer Normalization fused operation.
    Dequantize, add, layer-norm, then return the float output.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, scale, offset, gamma, beta):
        # Dequantize and add
        x_added = x1.float() * scale + offset + (x2.float() * scale + offset)

        # Layer normalization
        mean = x_added.mean(dim=-1, keepdim=True)
        var = x_added.var(dim=-1, keepdim=True, unbiased=False)
        output = ((x_added - mean) / torch.sqrt(var + self.epsilon)) * gamma + beta
        return output


def get_inputs():
    # Batch size: 32, Sequence length: 1024, Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    x1 = torch.randint(-128, 127, (batch_size, seq_len, hidden_size), dtype=torch.int8)
    x2 = torch.randint(-128, 127, (batch_size, seq_len, hidden_size), dtype=torch.int8)
    scale = torch.randn(1, dtype=torch.float32)
    offset = torch.randn(1, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)
    return [x1, x2, scale, offset, gamma, beta]


def get_init_inputs():
    return [1e-6]  # epsilon