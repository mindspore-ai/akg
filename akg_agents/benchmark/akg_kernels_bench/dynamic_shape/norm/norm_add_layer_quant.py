import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + Layer Normalization + Quantization fused operation.
    Add residual, layer-norm, then quantize.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma, beta, scale, zero_point):
        # Add residual connection
        x_added = x + residual

        # Layer normalization and affine
        mean = x_added.mean(dim=-1, keepdim=True)
        var = x_added.var(dim=-1, keepdim=True, unbiased=False)
        output = ((x_added - mean) / torch.sqrt(var + self.epsilon)) * gamma + beta

        # Quantize to int8
        output_quantized = torch.round(output / scale + zero_point).clamp(-128, 127).to(torch.int8)
        return output_quantized


def get_inputs_dyn_list():
    # Case 1: Small
    x1 = torch.randn(16, 64, 512, dtype=torch.float32)
    residual1 = torch.randn(16, 64, 512, dtype=torch.float32)
    gamma1 = torch.randn(512, dtype=torch.float32)
    beta1 = torch.randn(512, dtype=torch.float32)
    scale1 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point1 = torch.randn(1, dtype=torch.float32) * 10

    # Case 2: Middle
    x2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    residual2 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    gamma2 = torch.randn(4096, dtype=torch.float32)
    beta2 = torch.randn(4096, dtype=torch.float32)
    scale2 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point2 = torch.randn(1, dtype=torch.float32) * 10

    # Case 3: Large
    x3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    residual3 = torch.randn(64, 4096, 4096, dtype=torch.float32)
    gamma3 = torch.randn(4096, dtype=torch.float32)
    beta3 = torch.randn(4096, dtype=torch.float32)
    scale3 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point3 = torch.randn(1, dtype=torch.float32) * 10

    # Case 4: Non-aligned
    x4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    residual4 = torch.randn(48, 512, 2688, dtype=torch.float32)
    gamma4 = torch.randn(2688, dtype=torch.float32)
    beta4 = torch.randn(2688, dtype=torch.float32)
    scale4 = torch.randn(1, dtype=torch.float32) * 0.1 + 0.01
    zero_point4 = torch.randn(1, dtype=torch.float32) * 10

    return [
        [x1, residual1, gamma1, beta1, scale1, zero_point1],
        [x2, residual2, gamma2, beta2, scale2, zero_point2],
        [x3, residual3, gamma3, beta3, scale3, zero_point3],
        [x4, residual4, gamma4, beta4, scale4, zero_point4]
    ]

def get_init_inputs():
    return [1e-6]  # epsilon