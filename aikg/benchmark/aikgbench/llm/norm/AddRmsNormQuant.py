import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs AddRmsNormQuant operation.
    Based on 
    """

    def __init__(self, epsilon=1e-6, axis=-1, div_mode=False):
        super(Model, self).__init__()
        self.epsilon = epsilon
        self.axis = axis
        self.div_mode = div_mode

    def forward(self, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2):
        """
        Perform AddRmsNormQuant operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            gamma: Scale parameter tensor
            scales1: Quantization scales for first output
            scales2: Quantization scales for second output
            zero_points1: Quantization zero points for first output
            zero_points2: Quantization zero points for second output

        Returns:
            Tuple of (y1, y2, x) where:
            - y1, y2 are quantized outputs
            - x is the sum of input tensors
        """
        # Add the two input tensors
        x = x1 + x2

        # Compute RMS (Root Mean Square) normalization
        x_squared = x.pow(2)
        x_rms = torch.sqrt(x_squared.mean(
            dim=self.axis, keepdim=True) + self.epsilon)
        x_rstd = 1.0 / x_rms

        # Apply normalization
        x_normalized = x * x_rstd

        # Apply scale parameter
        output = x_normalized * gamma

        # Quantize outputs
        if self.div_mode:
            y1_quantized = torch.round(
                output / scales1 + zero_points1).clamp(-128, 127).to(torch.int8)
            y2_quantized = torch.round(
                output / scales2 + zero_points2).clamp(-128, 127).to(torch.int8)
        else:
            y1_quantized = torch.round(
                output * scales1 + zero_points1).clamp(-128, 127).to(torch.int8)
            y2_quantized = torch.round(
                output * scales2 + zero_points2).clamp(-128, 127).to(torch.int8)

        return y1_quantized, y2_quantized, x


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other rms norm operations
    batch_size, seq_len, hidden_size = 2, 1, 16

    # Generate input tensors (using float16 as specified in README)
    x1 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float16)

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float16)

    # Generate quantization parameters
    scales1 = torch.rand(1, dtype=torch.float32) * 0.1 + 0.01
    scales2 = torch.rand(1, dtype=torch.float32) * 0.1 + 0.01
    zero_points1 = torch.zeros(1, dtype=torch.int32)
    zero_points2 = torch.zeros(1, dtype=torch.int32)

    return [x1, x2, gamma, scales1, scales2, zero_points1, zero_points2]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-6, -1, False]  # epsilon=1e-6, axis=-1, div_mode=False
