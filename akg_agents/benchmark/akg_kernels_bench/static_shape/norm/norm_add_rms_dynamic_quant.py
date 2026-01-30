import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + RMS Normalization + Dynamic Quantization fused operation.
    This operation adds a residual connection to the input, applies RMS normalization,
    and then dynamically quantizes the result. Dynamic quantization computes quantization
    parameters on-the-fly based on the actual tensor values, which is more adaptive than
    static quantization.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma):
        """
        Perform Add + RMS Normalization + Dynamic Quantization fused operation.
        """
        # Add residual connection
        x_added = x + residual

        # Compute reciprocal RMS and normalize
        rstd = torch.rsqrt(x_added.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x_added * rstd) * gamma

        # Dynamic quantization: compute scale and zero_point from the tensor
        max_val = torch.abs(output).max()
        scale = max_val / 127.0  # Assuming int8 quantization
        zero_point = 0.0

        # Quantize to int8
        output_quantized = torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
        return output_quantized


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32
    # Sequence length: 1024
    # Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096

    # Generate input and residual tensors
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate scale parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [x, residual, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]