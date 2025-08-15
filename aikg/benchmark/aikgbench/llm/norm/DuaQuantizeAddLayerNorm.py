import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs DuaQuantizeAddLayerNorm operation.
    Based on 
    """

    def __init__(self, dtype=torch.float32, axis=-1, epsilon=1e-6, additional_output=False):
        super(Model, self).__init__()
        self.dtype = dtype
        self.axis = axis
        self.epsilon = epsilon
        self.additional_output = additional_output

    def forward(self, x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2):
        """
        Perform DuaQuantizeAddLayerNorm operation.

        Args:
            x1: First input tensor
            x2: Second input tensor
            gamma: Scale parameter tensor
            beta: Shift parameter tensor
            bias: Bias tensor
            scales1: Quantization scales for first output
            scales2: Quantization scales for second output
            zero_points1: Quantization zero points for first output
            zero_points2: Quantization zero points for second output

        Returns:
            Tuple of (y1, y2, x) where:
            - y1, y2 are quantized outputs
            - x is the sum of input tensors
        """
        # Add the two input tensors
        if bias is not None:
            x = x1 + x2 + bias
        else:
            x = x1 + x2

        # Get input shape and reshape for layer norm
        input_shape = x.shape
        row_size = x.shape[-1]
        row_count = 1
        for i in range(0, len(input_shape) - 1):
            row_count *= input_shape[i]

        x_shape = (row_count, row_size)
        x_mean_shape = (row_count, 1)

        # Reshape for layer norm computation
        x_reshaped = x.reshape(x_shape)

        # Compute mean and variance
        x_mean = torch.mean(x_reshaped, dim=1, keepdim=True)
        x_var = torch.var(x_reshaped, dim=1, keepdim=True,
                          unbiased=False) + self.epsilon
        x_rstd = 1.0 / torch.sqrt(x_var)

        # Broadcast tensors to match x_shape
        x_mean_broadcast = x_mean.expand(x_shape)
        x_rstd_broadcast = x_rstd.expand(x_shape)
        gamma_broadcast = gamma.expand(x_shape)
        beta_broadcast = beta.expand(x_shape)

        # Apply layer normalization
        y = torch.multiply(torch.multiply(
            x_reshaped - x_mean_broadcast, x_rstd_broadcast), gamma_broadcast) + beta_broadcast

        # Reshape back to original shape
        y = y.reshape(input_shape)

        # Quantize outputs
        y1_quantized = torch.round(
            y / scales1 + zero_points1).clamp(-128, 127).to(torch.int8)
        y2_quantized = torch.round(
            y / scales2 + zero_points2).clamp(-128, 127).to(torch.int8)

        return y1_quantized, y2_quantized, x


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use similar shapes as other layer norm operations
    batch_size, seq_len, hidden_size = 1, 2, 8

    # Generate input tensors
    x1 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    x2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate gamma and beta parameters
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)

    # Generate bias
    bias = torch.randn(hidden_size, dtype=torch.float32)

    # Generate quantization parameters
    scales1 = torch.rand(1, dtype=torch.float32) * 0.1 + 0.01
    scales2 = torch.rand(1, dtype=torch.float32) * 0.1 + 0.01
    zero_points1 = torch.zeros(1, dtype=torch.float32)
    zero_points2 = torch.zeros(1, dtype=torch.float32)

    return [x1, x2, gamma, beta, bias, scales1, scales2, zero_points1, zero_points2]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [torch.float32, -1, 1e-6, False]  # dtype=float32, axis=-1, epsilon=1e-6, additional_output=False
