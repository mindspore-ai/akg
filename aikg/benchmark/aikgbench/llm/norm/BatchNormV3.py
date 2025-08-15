import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs BatchNormV3 operation.
    Based on 
    """

    def __init__(self, momentum=0.1, epsilon=1e-5, is_training=True):
        super(Model, self).__init__()
        self.momentum = momentum
        self.epsilon = epsilon
        self.is_training = is_training

    def forward(self, x, weight, bias, running_mean, running_var):
        """
        Perform BatchNormV3 operation.

        Args:
            x: Input tensor
            weight: Weight parameter tensor
            bias: Bias parameter tensor
            running_mean: Running mean tensor
            running_var: Running variance tensor

        Returns:
            Tuple of (output, running_mean, running_var, save_mean, save_rstd)
        """
        # Use PyTorch's native batch norm
        output, save_mean, save_rstd = torch.ops.aten.native_batch_norm(
            input=x,
            weight=weight,
            bias=bias,
            running_mean=running_mean,
            running_var=running_var,
            training=self.is_training,
            momentum=self.momentum,
            eps=self.epsilon
        )

        return output, running_mean, running_var, save_mean, save_rstd


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, channels, height, width = 1, 2, 1, 4

    # Generate input tensor
    x = torch.randn(batch_size, channels, height, width, dtype=torch.float32)

    # Generate weight and bias parameters
    weight = torch.ones(channels, dtype=torch.float32)
    bias = torch.zeros(channels, dtype=torch.float32)

    # Generate running statistics
    running_mean = torch.zeros(channels, dtype=torch.float32)
    running_var = torch.ones(channels, dtype=torch.float32)

    return [x, weight, bias, running_mean, running_var]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [0.1, 1e-5, True]  # momentum=0.1, epsilon=1e-5, is_training=True
