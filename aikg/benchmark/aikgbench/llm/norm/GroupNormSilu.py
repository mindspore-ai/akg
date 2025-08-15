import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GroupNormSilu operation.
    Based on 
    """

    def __init__(self, num_groups=2, epsilon=1e-6, activate_silu=True):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.activate_silu = activate_silu

    def forward(self, x, gamma, beta):
        """
        Perform GroupNormSilu operation.

        Args:
            x: Input tensor
            gamma: Scale parameter tensor
            beta: Shift parameter tensor

        Returns:
            Tuple of (output, mean_out, rstd_out) where:
            - output is the normalized and silu-activated tensor
            - mean_out is the mean of each group
            - rstd_out is the reciprocal standard deviation of each group
        """
        # Get input dimensions
        N, C = x.shape[:2]
        remaining_dims = x.shape[2:]
        HxW = 1
        for size in remaining_dims:
            HxW *= size

        # Use PyTorch's native group norm
        output, mean_out, rstd_out = torch.ops.aten.native_group_norm(
            input=x,
            weight=gamma,
            bias=beta,
            N=N,
            C=C,
            HxW=HxW,
            group=self.num_groups,
            eps=self.epsilon
        )

        # Apply SiLU activation if enabled
        if self.activate_silu:
            sigmoid_out = 1 / (1 + torch.exp(-output))
            output = output * sigmoid_out

        return output, mean_out, rstd_out


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, channels, height, width = 4, 2, 8, 8

    # Generate input tensor
    x = torch.rand(batch_size, channels, height, width,
                   dtype=torch.float32) * 0.9 + 0.1

    # Generate gamma and beta parameters
    gamma = torch.rand(channels, dtype=torch.float32) * 0.9 + 0.1
    beta = torch.rand(channels, dtype=torch.float32) * 0.9 + 0.1

    return [x, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [2, 1e-6, True]  # num_groups=2, epsilon=1e-6, activate_silu=True
