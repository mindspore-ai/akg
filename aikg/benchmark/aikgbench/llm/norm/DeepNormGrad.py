import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs DeepNormGrad operation.
    Based on 
    """

    def __init__(self, alpha=0.3):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, dy, x, gx, gamma, mean, rstd):
        """
        Perform DeepNormGrad operation (backward pass).

        Args:
            dy: Gradient of output
            x: Input tensor from forward pass
            gx: Gate tensor from forward pass
            gamma: Scale parameter tensor
            mean: Mean from forward pass
            rstd: Reciprocal standard deviation from forward pass

        Returns:
            Tuple of (dx, dgx, dbeta, dgamma) where:
            - dx is the gradient with respect to input x
            - dgx is the gradient with respect to gate gx
            - dbeta is the gradient with respect to beta
            - dgamma is the gradient with respect to gamma
        """
        # Apply alpha scaling and add gate (same as forward pass)
        x_add = x * self.alpha + gx

        # Compute normalized output (same as forward pass)
        diff = x_add - mean
        output = gamma * diff * rstd

        # Compute gradients for DeepNorm backward
        # This is a simplified implementation of the backward pass

        # Gradient with respect to gamma
        dgamma = torch.sum(dy * diff * rstd, dim=0)

        # Gradient with respect to beta (simplified)
        dbeta = torch.sum(dy, dim=0)

        # Gradient with respect to normalized output
        dy_normalized = dy * gamma * rstd

        # Gradient with respect to x_add
        dx_add = dy_normalized

        # Gradient with respect to x and gx
        dx = dx_add * self.alpha
        dgx = dx_add

        return dx, dgx, dbeta, dgamma


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other deep norm operations
    batch_size, seq_len, hidden_size = 3, 1, 4

    # Generate input tensors
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     dtype=torch.float32).reshape(batch_size, seq_len, hidden_size)
    gx = torch.tensor([2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
                      dtype=torch.float32).reshape(batch_size, seq_len, hidden_size)

    # Generate gradient of output
    dy = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate statistics from forward pass
    alpha = 0.3
    x_add = x * alpha + gx
    mean = x_add.mean(-1, keepdim=True)
    diff = x_add - mean
    variance = diff.pow(2).mean(-1, keepdim=True) + 1e-6
    rstd = 1.0 / torch.sqrt(variance)

    # Generate gamma parameter
    gamma = torch.tensor([0, 1, 2, 3], dtype=torch.float32)

    return [dy, x, gx, gamma, mean, rstd]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [0.3]  # alpha=0.3
