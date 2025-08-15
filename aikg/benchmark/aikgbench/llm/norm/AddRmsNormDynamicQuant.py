import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs AddRmsNormDynamicQuant operation.
    Based on 
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, gamma, smooth_scale1, smooth_scale2):
        """
        Perform AddRmsNormDynamicQuant operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            gamma: Scale parameter tensor
            smooth_scale1: Smooth scale for first output
            smooth_scale2: Smooth scale for second output

        Returns:
            Tuple of (y1, y2, x, scale1, scale2) where:
            - y1, y2 are dynamically quantized outputs
            - x is the sum of input tensors
            - scale1, scale2 are dynamic quantization scales
        """
        # Add the two input tensors
        x = x1 + x2

        # Compute RMS (Root Mean Square) normalization
        x_squared = x.pow(2)
        x_rms = torch.sqrt(x_squared.mean(dim=-1, keepdim=True) + self.epsilon)
        x_rstd = 1.0 / x_rms

        # Apply normalization
        x_normalized = x * x_rstd

        # Apply scale parameter
        output = x_normalized * gamma

        # Dynamic quantization for first output
        max_val1 = torch.max(torch.abs(output))
        scale1 = max_val1 / 127.0
        y1 = torch.round(output / scale1).clamp(-128, 127).to(torch.int8)

        # Dynamic quantization for second output
        max_val2 = torch.max(torch.abs(output))
        scale2 = max_val2 / 127.0
        y2 = torch.round(output / scale2).clamp(-128, 127).to(torch.int8)

        return y1, y2, x, scale1, scale2


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

    # Generate smooth scales
    smooth_scale1 = torch.rand(1, dtype=torch.float16) * 0.1 + 0.01
    smooth_scale2 = torch.rand(1, dtype=torch.float16) * 0.1 + 0.01

    return [x1, x2, gamma, smooth_scale1, smooth_scale2]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-6]  # epsilon=1e-6
