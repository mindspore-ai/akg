import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs RmsNormGrad operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy, x, rstd, gamma):
        """
        Perform RmsNormGrad operation (backward pass).

        Args:
            dy: Gradient of output
            x: Input tensor from forward pass
            rstd: Reciprocal standard deviation from forward pass
            gamma: Scale parameter tensor

        Returns:
            Tuple of (dx, dgamma) where:
            - dx is the gradient with respect to input
            - dgamma is the gradient with respect to gamma
        """
        # Compute gradients for RMS norm backward
        # This is a simplified implementation of the backward pass

        # Gradient with respect to gamma
        dgamma = torch.sum(dy * x * rstd, dim=0)

        # Gradient with respect to input
        dx = dy * gamma * rstd

        return dx, dgamma


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other rms norm operations
    batch_size, seq_len, hidden_size = 2, 1, 16

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gradient of output
    dy = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate statistics from forward pass
    x_squared = x.pow(2)
    x_rms = torch.sqrt(x_squared.mean(dim=-1, keepdim=True) + 1e-6)
    rstd = 1.0 / x_rms

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [dy, x, rstd, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    For gradient operations, no specific initialization parameters are needed.
    """
    return []
