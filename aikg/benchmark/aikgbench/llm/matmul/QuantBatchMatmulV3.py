import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs QuantBatchMatmulV3 operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, scale, offset, bias=None):
        """
        Perform QuantBatchMatmulV3 operation.

        Args:
            x1: First quantized input tensor
            x2: Second quantized input tensor
            scale: Scale factor for quantization
            offset: Offset factor for quantization
            bias: Optional bias tensor

        Returns:
            Quantized output tensor
        """
        # Convert to float for matrix multiplication
        x1_float = x1.float()
        x2_float = x2.float()

        # Perform matrix multiplication
        output = torch.matmul(x1_float, x2_float)

        # Add bias if provided
        if bias is not None:
            output = output + bias

        # Apply scale and offset
        output = output * scale + offset

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from README: x1(16, 32), x2(32, 16)
    m, k, n = 16, 32, 16

    # Generate quantized input tensors (using int8 as specified)
    x1 = torch.randint(-128, 127, (m, k), dtype=torch.int8)
    x2 = torch.randint(-128, 127, (k, n), dtype=torch.int8)

    # Generate scale and offset
    scale = torch.randn(1, dtype=torch.float32)
    offset = torch.randn(1, dtype=torch.float32)

    # Generate optional bias
    bias = torch.randn(n, dtype=torch.float32)

    return [x1, x2, scale, offset, bias]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
