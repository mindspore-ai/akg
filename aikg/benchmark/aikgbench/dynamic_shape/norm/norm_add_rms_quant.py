import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + RMS Normalization + Quantization fused operation.
    This operation adds a residual connection to the input, applies RMS normalization,
    and then quantizes the result. This fusion is commonly used in quantized neural networks
    to reduce memory usage while maintaining numerical stability through normalization.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma, scale, zero_point):
        """
        Perform Add + RMS Normalization + Quantization fused operation.
        """
        # Add residual connection
        x_added = x + residual

        # Compute reciprocal RMS and normalize
        rstd = torch.rsqrt(x_added.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x_added * rstd) * gamma

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
    scale1 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point1 = torch.randn(1, dtype=torch.float32) * 10

    # Case 2: Middle
    x2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    residual2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    gamma2 = torch.randn(4096, dtype=torch.float32)
    scale2 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point2 = torch.randn(1, dtype=torch.float32) * 10

    # Case 3: Large
    x3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    residual3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    gamma3 = torch.randn(4096, dtype=torch.float32)
    scale3 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point3 = torch.randn(1, dtype=torch.float32) * 10

    # Case 4: Non-aligned
    x4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    residual4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    gamma4 = torch.randn(2688, dtype=torch.float32)
    scale4 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point4 = torch.randn(1, dtype=torch.float32) * 10

    return [
        [x1, residual1, gamma1, scale1, zero_point1],
        [x2, residual2, gamma2, scale2, zero_point2],
        [x3, residual3, gamma3, scale3, zero_point3],
        [x4, residual4, gamma4, scale4, zero_point4]
    ]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]