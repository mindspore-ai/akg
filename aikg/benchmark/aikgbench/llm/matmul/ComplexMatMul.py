import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs ComplexMatMul operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, bias):
        """
        Perform ComplexMatMul operation.

        Args:
            x: First complex input tensor
            y: Second complex input tensor
            bias: Complex bias tensor

        Returns:
            Complex output tensor after matrix multiplication
        """
        # Perform complex matrix multiplication
        output = torch.matmul(x, y)

        # Add bias
        output = output + bias

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use complex64 data type as specified in README
    batch_size, m, k, n = 2, 16, 32, 16

    # Generate complex input tensors
    x_real = torch.randn(batch_size, m, k, dtype=torch.float32)
    x_imag = torch.randn(batch_size, m, k, dtype=torch.float32)
    x = torch.complex(x_real, x_imag)

    y_real = torch.randn(batch_size, k, n, dtype=torch.float32)
    y_imag = torch.randn(batch_size, k, n, dtype=torch.float32)
    y = torch.complex(y_real, y_imag)

    # Generate complex bias
    bias_real = torch.randn(batch_size, m, n, dtype=torch.float32)
    bias_imag = torch.randn(batch_size, m, n, dtype=torch.float32)
    bias = torch.complex(bias_real, bias_imag)

    return [x, y, bias]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
