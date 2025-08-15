import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GroupNormGrad operation.
    Based on 
    """

    def __init__(self, num_groups=2, data_format="NCHW", dx_is_require=True, dgamma_is_require=True, dbeta_is_require=True):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format
        self.dx_is_require = dx_is_require
        self.dgamma_is_require = dgamma_is_require
        self.dbeta_is_require = dbeta_is_require

    def forward(self, dy, mean, rstd, x, gamma):
        """
        Perform GroupNormGrad operation (backward pass).

        Args:
            dy: Gradient of output
            mean: Mean from forward pass
            rstd: Reciprocal standard deviation from forward pass
            x: Input tensor from forward pass
            gamma: Scale parameter tensor

        Returns:
            Tuple of (dx, dgamma, dbeta) where:
            - dx is the gradient with respect to input
            - dgamma is the gradient with respect to gamma
            - dbeta is the gradient with respect to beta
        """
        # Get input dimensions
        N, C = x.shape[:2]
        remaining_dims = x.shape[2:]
        HxW = 1
        for size in remaining_dims:
            HxW *= size

        # Reshape for group norm computation
        x_reshaped = x.reshape(N, self.num_groups, C // self.num_groups, HxW)
        dy_reshaped = dy.reshape(N, self.num_groups, C // self.num_groups, HxW)

        # Compute gradients for group norm backward
        # This is a simplified implementation of the backward pass
        if self.dgamma_is_require:
            dgamma = torch.sum(dy * (x - mean) * rstd, dim=0)
        else:
            dgamma = torch.zeros_like(gamma)

        if self.dbeta_is_require:
            dbeta = torch.sum(dy, dim=0)
        else:
            dbeta = torch.zeros_like(gamma)

        # Compute gradient with respect to input
        if self.dx_is_require:
            dx = dy * gamma * rstd
        else:
            dx = torch.zeros_like(x)

        return dx, dgamma, dbeta


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other group norm operations
    batch_size, channels, height, width = 4, 2, 8, 8

    # Generate input tensor
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

    # Generate gradient of output
    dy = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

    # Generate statistics from forward pass
    N, C = x.shape[:2]
    remaining_dims = x.shape[2:]
    HxW = 1
    for size in remaining_dims:
        HxW *= size

    x_reshaped = x.reshape(N, 2, C // 2, HxW)  # num_groups=2
    mean = torch.mean(x_reshaped, dim=(2, 3), keepdim=True)
    var = torch.var(x_reshaped, dim=(2, 3),
                    keepdim=True, unbiased=False) + 1e-6
    rstd = 1.0 / torch.sqrt(var)

    # Reshape back to original shape
    mean = mean.reshape(x.shape)
    rstd = rstd.reshape(x.shape)

    # Generate gamma parameter
    gamma = torch.randn(channels, dtype=torch.float32)

    return [dy, mean, rstd, x, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [2, "NCHW", True, True, True]  # num_groups=2, data_format="NCHW", dx_is_require=True, dgamma_is_require=True, dbeta_is_require=True
