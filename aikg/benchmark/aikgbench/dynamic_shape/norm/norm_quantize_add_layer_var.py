import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Quantized Add + Layer Normalization fused operation.
    This operation adds two quantized tensors, dequantizes the result, and then applies layer normalization.
    This fusion is commonly used in quantized neural networks to reduce memory usage while
    maintaining numerical stability through normalization.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x1, x2, scale, offset, gamma, beta):
        """
        Perform Quantized Add + Layer Normalization fused operation.

        Args:
            x1: First quantized input tensor of shape (batch_size, seq_len, hidden_size) with int8 dtype
            x2: Second quantized input tensor of shape (batch_size, seq_len, hidden_size) with int8 dtype
            scale: Scale factor for dequantization (float32)
            offset: Offset factor for dequantization (float32)
            gamma: Scale parameter for layer norm of shape (hidden_size,) with float32 dtype
            beta: Shift parameter for layer norm of shape (hidden_size,) with float32 dtype

        Returns:
            Output tensor after dequantization and layer normalization
        """
        # Dequantize both input tensors
        x1_float = x1.float() * scale + offset
        x2_float = x2.float() * scale + offset

        # Add the dequantized tensors
        x_added = x1_float + x2_float

        # Layer normalization with efficient computation
        mean = x_added.mean(dim=-1, keepdim=True)
        var = x_added.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + self.epsilon)
        x_normalized = (x_added - mean) * rstd

        # Apply scale and shift parameters
        output = x_normalized * gamma + beta

        return output

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    batch_size1, seq_len1, hidden_size1 = 16, 512, 2048
    x1_1 = torch.randint(-128, 127, (batch_size1, seq_len1, hidden_size1), dtype=torch.int8)
    x2_1 = torch.randint(-128, 127, (batch_size1, seq_len1, hidden_size1), dtype=torch.int8)
    scale1 = torch.randn(1, dtype=torch.float32)
    offset1 = torch.randn(1, dtype=torch.float32)
    gamma1 = torch.randn(hidden_size1, dtype=torch.float32)
    beta1 = torch.randn(hidden_size1, dtype=torch.float32)

    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    batch_size2, seq_len2, hidden_size2 = 24, 768, 3072
    x1_2 = torch.randint(-128, 127, (batch_size2, seq_len2, hidden_size2), dtype=torch.int8)
    x2_2 = torch.randint(-128, 127, (batch_size2, seq_len2, hidden_size2), dtype=torch.int8)
    scale2 = torch.randn(1, dtype=torch.float32)
    offset2 = torch.randn(1, dtype=torch.float32)
    gamma2 = torch.randn(hidden_size2, dtype=torch.float32)
    beta2 = torch.randn(hidden_size2, dtype=torch.float32)

    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    batch_size3, seq_len3, hidden_size3 = 32, 1024, 4096
    x1_3 = torch.randint(-128, 127, (batch_size3, seq_len3, hidden_size3), dtype=torch.int8)
    x2_3 = torch.randint(-128, 127, (batch_size3, seq_len3, hidden_size3), dtype=torch.int8)
    scale3 = torch.randn(1, dtype=torch.float32)
    offset3 = torch.randn(1, dtype=torch.float32)
    gamma3 = torch.randn(hidden_size3, dtype=torch.float32)
    beta3 = torch.randn(hidden_size3, dtype=torch.float32)

    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    batch_size4, seq_len4, hidden_size4 = 48, 1536, 6144
    x1_4 = torch.randint(-128, 127, (batch_size4, seq_len4, hidden_size4), dtype=torch.int8)
    x2_4 = torch.randint(-128, 127, (batch_size4, seq_len4, hidden_size4), dtype=torch.int8)
    scale4 = torch.randn(1, dtype=torch.float32)
    offset4 = torch.randn(1, dtype=torch.float32)
    gamma4 = torch.randn(hidden_size4, dtype=torch.float32)
    beta4 = torch.randn(hidden_size4, dtype=torch.float32)

    return [
        [x1_1, x2_1, scale1, offset1, gamma1, beta1],
        [x1_2, x2_2, scale2, offset2, gamma2, beta2],
        [x1_3, x2_3, scale3, offset3, gamma3, beta3],
        [x1_4, x2_4, scale4, offset4, gamma4, beta4]
    ]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]