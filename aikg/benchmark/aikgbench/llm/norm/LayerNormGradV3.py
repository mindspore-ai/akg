import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs LayerNormGradV3 operation.
    Based on 
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy, x, rstd, mean, gamma):
        """
        Perform LayerNormGradV3 operation (backward pass).

        Args:
            dy: Gradient of output
            x: Input tensor from forward pass
            rstd: Reciprocal standard deviation from forward pass
            mean: Mean from forward pass
            gamma: Scale parameter tensor

        Returns:
            Tuple of (pd_x, pd_gamma, pd_beta) where:
            - pd_x is the gradient with respect to input
            - pd_gamma is the gradient with respect to gamma
            - pd_beta is the gradient with respect to beta
        """
        # Compute gradients for layer norm backward
        # This is a simplified implementation of the backward pass

        # Gradient with respect to gamma
        pd_gamma = torch.sum(dy * (x - mean) * rstd, dim=0)

        # Gradient with respect to beta
        pd_beta = torch.sum(dy, dim=0)

        # Gradient with respect to input
        pd_x = dy * gamma * rstd

        return pd_x, pd_gamma, pd_beta


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other layer norm operations
    batch_size, seq_len, hidden_size = 1, 2, 32

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gradient of output
    dy = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate statistics from forward pass
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
    rstd = 1.0 / torch.sqrt(var)

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [dy, x, rstd, mean, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    For gradient operations, no specific initialization parameters are needed.
    """
    return []
