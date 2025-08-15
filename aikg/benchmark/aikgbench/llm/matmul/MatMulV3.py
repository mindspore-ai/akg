import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs MatMulV3 operation.
    """

    def __init__(self, transpose_x1=False, transpose_x2=False, offset_x=0, enable_hf32=False):
        super(Model, self).__init__()
        self.transpose_x1 = transpose_x1
        self.transpose_x2 = transpose_x2
        self.offset_x = offset_x
        self.enable_hf32 = enable_hf32

    def forward(self, x1, x2, bias=None, offset_w=None):
        """
        Perform MatMulV3 operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            bias: Optional bias tensor
            offset_w: Optional offset tensor

        Returns:
            Output tensor after matrix multiplication
        """
        # Apply transpose operations if needed
        if self.transpose_x1:
            x1 = x1.transpose(-2, -1)
        if self.transpose_x2:
            x2 = x2.transpose(-2, -1)

        # Perform matrix multiplication
        output = torch.matmul(x1, x2)

        # Add bias if provided
        if bias is not None:
            output = output + bias

        # Add offset if provided
        if offset_w is not None:
            output = output + offset_w

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from README: x1(2, 16, 32), x2(2, 32, 16)
    batch_size, m, k, n = 2, 16, 32, 16

    # Generate input tensors
    x1 = torch.randn(batch_size, m, k, dtype=torch.float32)
    x2 = torch.randn(batch_size, k, n, dtype=torch.float32)

    # Generate optional bias
    bias = torch.randn(batch_size, 1, n, dtype=torch.float32)

    # Generate optional offset
    offset_w = torch.randn(batch_size, m, n, dtype=torch.float32)

    return [x1, x2, bias, offset_w]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [False, False, 0, False]  # transpose_x1=False, transpose_x2=False, offset_x=0, enable_hf32=False
