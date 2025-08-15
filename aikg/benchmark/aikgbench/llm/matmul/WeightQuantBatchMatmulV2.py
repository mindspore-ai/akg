import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs WeightQuantBatchMatmulV2 operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, weight, antiquantScale, antiquantOffsetOptional=None, quantScaleOptional=None, quantOffsetOptional=None):
        """
        Perform WeightQuantBatchMatmulV2 operation.

        Args:
            x: Input tensor
            weight: Quantized weight tensor
            antiquantScale: Scale for weight dequantization
            antiquantOffsetOptional: Optional offset for weight dequantization
            quantScaleOptional: Optional scale for output quantization
            quantOffsetOptional: Optional offset for output quantization

        Returns:
            Output tensor after weight dequantization and matrix multiplication
        """
        # Dequantize weight: ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
        dequantized_weight = weight.float()
        if antiquantOffsetOptional is not None:
            dequantized_weight = dequantized_weight + antiquantOffsetOptional
        dequantized_weight = dequantized_weight * antiquantScale

        # Convert x to float32 for matrix multiplication if needed
        x_float = x.float() if x.dtype != torch.float32 else x

        # Perform matrix multiplication
        output = torch.matmul(x_float, dequantized_weight)

        # Apply output quantization if provided
        if quantScaleOptional is not None:
            output = output * quantScaleOptional
            if quantOffsetOptional is not None:
                output = output + quantOffsetOptional

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from README: x(16, 32), weight(32, 16)
    m, k, n = 16, 32, 16

    # Generate input tensors (using float16 as specified)
    x = torch.randn(m, k, dtype=torch.float16)

    # Generate quantized weight (using int8 as specified)
    weight = torch.randint(-128, 127, (k, n), dtype=torch.int8)

    # Generate dequantization parameters
    antiquantScale = torch.randn(1, dtype=torch.float16)
    antiquantOffsetOptional = torch.randn(1, dtype=torch.float16)

    # Generate optional quantization parameters
    quantScaleOptional = torch.randn(1, dtype=torch.float32)
    quantOffsetOptional = torch.randn(1, dtype=torch.float32)

    return [x, weight, antiquantScale, antiquantOffsetOptional, quantScaleOptional, quantOffsetOptional]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
