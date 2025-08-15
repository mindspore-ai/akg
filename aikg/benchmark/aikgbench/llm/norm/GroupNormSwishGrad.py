import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GroupNormSwishGrad operation.
    Based on 
    """

    def __init__(self, num_groups=2, data_format="NCHW", swish_scale=1.0, dgamma_is_require=True, dbeta_is_require=True):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.data_format = data_format
        self.swish_scale = swish_scale
        self.dgamma_is_require = dgamma_is_require
        self.dbeta_is_require = dbeta_is_require

    def forward(self, dy, mean, rstd, x, gamma, beta):
        """
        Perform GroupNormSwishGrad operation (backward pass).

        Args:
            dy: Gradient of output
            mean: Mean from forward pass
            rstd: Reciprocal standard deviation from forward pass
            x: Input tensor from forward pass
            gamma: Scale parameter tensor
            beta: Shift parameter tensor

        Returns:
            Tuple of (dx_out, dgamma_out, dbeta_out) where:
            - dx_out is the gradient with respect to input
            - dgamma_out is the gradient with respect to gamma
            - dbeta_out is the gradient with respect to beta
        """
        # Get input dimensions
        N, C = x.shape[:2]
        remaining_dims = x.shape[2:]
        HxW = 1
        for size in remaining_dims:
            HxW *= size

        # Reshape for group norm computation
        x_reshaped = x.reshape(N, self.num_groups, C // self.num_groups, HxW)

        # Apply group normalization
        x_normalized = (x_reshaped - mean) * rstd
        output = x_normalized * gamma + beta

        # Apply Swish activation
        sigmoid_out = 1 / (1 + torch.exp(-self.swish_scale * output))
        swish_output = output * sigmoid_out

        # Compute gradients for Swish backward
        # d(swish_output)/d(output) = sigmoid_out + output * sigmoid_out * (1 - sigmoid_out) * swish_scale
        swish_grad = sigmoid_out + swish_output * \
            (1 - sigmoid_out) * self.swish_scale

        # Apply gradient through Swish
        dy_swish = dy * swish_grad

        # Compute gradients for group norm backward
        if self.dgamma_is_require:
            dgamma_out = torch.sum(dy_swish * x_normalized, dim=0)
        else:
            dgamma_out = torch.zeros_like(gamma)

        if self.dbeta_is_require:
            dbeta_out = torch.sum(dy_swish, dim=0)
        else:
            dbeta_out = torch.zeros_like(beta)

        # Compute gradient with respect to input
        dx_out = dy_swish * gamma * rstd

        return dx_out, dgamma_out, dbeta_out


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

    # Generate gamma and beta parameters
    gamma = torch.randn(channels, dtype=torch.float32)
    beta = torch.randn(channels, dtype=torch.float32)

    return [dy, mean, rstd, x, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [2, "NCHW", 1.0, True, True]  # num_groups=2, data_format="NCHW", swish_scale=1.0, dgamma_is_require=True, dbeta_is_require=True
