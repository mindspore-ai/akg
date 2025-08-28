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
        """
        Perform Quantized Add + Layer Normalization fused operation.
        
        Args:
            x1: First quantized input tensor of shape (batch_size, seq_len, hidden_size)
            x2: Second quantized input tensor of shape (batch_size, seq_len, hidden_size)
            scale: Scale factor for dequantization
            offset: Offset factor for dequantization
            gamma: Scale parameter for layer norm
            beta: Shift parameter for layer norm
            
        Returns:
            Output tensor after dequantization and layer normalization
        """
        # Dequantize both input tensors
        x1_float = x1.float() * scale + offset
        x2_float = x2.float() * scale + offset
        
        # Add the dequantized tensors
        x_added = x1_float + x2_float

        # Layer normalization with efficient computation
        mean = x_added.mean(dim=-1, keepdim=True)
        var = x_added.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + self.epsilon)
        x_normalized = (x_added - mean) * rstd
        
        # Apply scale and shift parameters
        output = x_normalized * gamma + beta
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