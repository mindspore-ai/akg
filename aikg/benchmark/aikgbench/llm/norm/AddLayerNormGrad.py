import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs AddLayerNormGrad operation. 
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy, x1, x2, rstd, mean, gamma, dsum):
        """
        Perform AddLayerNormGrad operation (backward pass).

        Args:
            dy: Gradient of output
            x1: First input tensor
            x2: Second input tensor
            rstd: Reciprocal standard deviation from forward pass
            mean: Mean from forward pass
            gamma: Scale parameter tensor
            dsum: Sum of gradients

        Returns:
            Tuple of (dx, dgamma, dbeta) where:
            - dx is the gradient with respect to input
            - dgamma is the gradient with respect to gamma
            - dbeta is the gradient with respect to beta
        """
        # Add the two input tensors (same as forward pass)
        x = x1 + x2

        # Compute gradients for layer norm backward
        # This is a simplified implementation of the backward pass
        N = x.shape[0]
        C = x.shape[-1]

        # Compute gradients for gamma and beta
        dgamma = torch.sum(dy * (x - mean) * rstd, dim=0)
        dbeta = torch.sum(dy, dim=0)

        # Compute gradient with respect to input
        # Simplified gradient computation
        dx = dy * gamma * rstd

        return dx, dgamma, dbeta


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other layer norm operations
    batch_size, seq_len, hidden_size = 2, 1, 16

    # Generate input tensors
    x1 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gradient of output
    dy = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate statistics from forward pass
    x = x1 + x2
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-6
    rstd = 1.0 / torch.sqrt(var)

    # Generate gamma parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    # Generate dsum (simplified)
    dsum = torch.randn_like(dy)

    return [dy, x1, x2, rstd, mean, gamma, dsum]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    For gradient operations, no specific initialization parameters are needed.
    """
    return []
