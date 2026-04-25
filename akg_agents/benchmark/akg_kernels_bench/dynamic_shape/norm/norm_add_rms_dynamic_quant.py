import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + RMS Normalization + Dynamic Quantization fused operation.
    This operation adds a residual connection to the input, applies RMS normalization,
    and then dynamically quantizes the result. Dynamic quantization computes quantization
    parameters on-the-fly based on the actual tensor values, which is more adaptive than
    static quantization.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma):
        """
        Perform Add + RMS Normalization + Dynamic Quantization fused operation.
        """
        # Add residual connection
        x_added = x + residual

        # Compute reciprocal RMS and normalize
        rstd = torch.rsqrt(x_added.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x_added * rstd) * gamma

        # Dynamic quantization: compute scale and zero_point from the tensor
        max_val = torch.abs(output).max()
        scale = max_val / 127.0  # Assuming int8 quantization
        zero_point = 0.0

        # Quantize to int8
        output_quantized = torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
        return output_quantized


def get_inputs_dyn_list():
    """
    Generate random input tensors for testing with different model shapes.
    """
    # Case 1: Small
    x1 = torch.randn(16, 64, 512, dtype=torch.float32)
    residual1 = torch.randn(16, 64, 512, dtype=torch.float32)
    gamma1 = torch.randn(512, dtype=torch.float32)

    # Case 2: Middle
    x2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    residual2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    gamma2 = torch.randn(4096, dtype=torch.float32)

    # Case 3: Large
    x3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    residual3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    gamma3 = torch.randn(4096, dtype=torch.float32)

    # Case 4: Non-aligned
    x4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    residual4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    gamma4 = torch.randn(2688, dtype=torch.float32)

    return [
        [x1, residual1, gamma1],
        [x2, residual2, gamma2],
        [x3, residual3, gamma3],
        [x4, residual4, gamma4]
    ]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]