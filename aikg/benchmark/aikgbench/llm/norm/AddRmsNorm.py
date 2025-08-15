import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs AddRmsNorm operation.
    Based on 
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, gamma):
        """
        Perform AddRmsNorm operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            gamma: Scale parameter tensor

        Returns:
            Tuple of (output, rstd, x) where:
            - output is the normalized tensor
            - rstd is the reciprocal standard deviation
            - x is the sum of input tensors
        """
        # Add the two input tensors
        x = x1 + x2

        # Compute RMS (Root Mean Square) normalization
        # Unlike LayerNorm, RmsNorm doesn't subtract the mean
        x_squared = x.pow(2)
        x_rms = torch.sqrt(x_squared.mean(dim=-1, keepdim=True) + self.epsilon)
        x_rstd = 1.0 / x_rms

        # Apply normalization
        x_normalized = x * x_rstd

        # Apply scale parameter
        output = x_normalized * gamma

        return output, x_rstd, x


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, seq_len, hidden_size = 2, 1, 16

    # Generate input tensors
    x1 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [x1, x2, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-6]  # epsilon=1e-6
