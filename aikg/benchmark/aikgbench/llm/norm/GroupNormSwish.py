import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GroupNormSwish operation.
    Based on 
    """

    def __init__(self, num_groups=8, epsilon=1e-5, activate_swish=True, swish_scale=1.0):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.epsilon = epsilon
        self.activate_swish = activate_swish
        self.swish_scale = swish_scale

    def forward(self, x, gamma, beta):
        """
        Perform GroupNormSwish operation.

        Args:
            x: Input tensor
            gamma: Scale parameter tensor
            beta: Shift parameter tensor

        Returns:
            Tuple of (output, mean_out, rstd_out) where:
            - output is the normalized and swish-activated tensor
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

        # Apply Swish activation if enabled
        if self.activate_swish:
            sigmoid_out = 1 / (1 + torch.exp(-self.swish_scale * output))
            output = output * sigmoid_out

        return output, mean_out, rstd_out


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    N, C = 100, 32
    x_shape = (N, C)

    # Generate input tensor
    x = torch.rand(x_shape, dtype=torch.float16)

    # Generate gamma and beta parameters
    gamma = torch.rand(C, dtype=torch.float16)
    beta = torch.rand(C, dtype=torch.float16)

    return [x, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [8, 1e-5, True, 1.0]  # num_groups=8, epsilon=1e-5, activate_swish=True, swish_scale=1.0
