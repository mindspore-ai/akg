import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs AddRmsNormCast operation.
    Based on 
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, gamma):
        """
        Perform AddRmsNormCast operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            gamma: Scale parameter tensor

        Returns:
            Tuple of (y1, y2, rstd, x) where:
            - y1 is the cast output (float)
            - y2 is the normalized output (original dtype)
            - rstd is the reciprocal standard deviation
            - x is the sum of input tensors
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

        # Cast to float for y1
        y1 = output.to(torch.float32)

        # Keep original dtype for y2
        y2 = output

        return y1, y2, x_rstd, x


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

    return [x1, x2, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-6]  # epsilon=1e-6
