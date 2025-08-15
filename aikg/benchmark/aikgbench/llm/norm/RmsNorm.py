import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs RmsNorm operation.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, gamma):
        """
        Perform RmsNorm operation.

        Args:
            x: Input tensor
            gamma: Scale parameter tensor

        Returns:
            Tuple of (output, rstd) where output is the normalized tensor
        """
        # Compute RMS (Root Mean Square) normalization
        # Unlike LayerNorm, RmsNorm doesn't subtract the mean
        x_squared = x.pow(2)
        x_rms = torch.sqrt(x_squared.mean(dim=-1, keepdim=True) + self.epsilon)
        x_rstd = 1.0 / x_rms

        # Apply normalization
        x_normalized = x * x_rstd

        # Apply scale parameter
        output = x_normalized * gamma

        return output, x_rstd


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use the same shapes as in gen_data.py
    batch_size, seq_len, hidden_size = 2, 1, 16

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [x, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [1e-6]  # epsilon=1e-6
