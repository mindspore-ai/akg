import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + RMS Normalization + Cast fused operation.
    This operation adds a residual connection to the input, applies RMS normalization,
    and then casts the result to a different data type. The cast operation is useful
    for reducing memory bandwidth requirements and improving computational efficiency.
    """

    def __init__(self, epsilon=1e-6, target_dtype=torch.float16):
        super(Model, self).__init__()
        self.epsilon = epsilon
        self.target_dtype = target_dtype

    def forward(self, x, residual, gamma):
        """
        Perform Add + RMS Normalization + Cast fused operation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            residual: Residual tensor of shape (batch_size, seq_len, hidden_size)
            gamma: Scale parameter of shape (hidden_size,)

        Returns:
            Output tensor after add + RMS norm + cast
        """
        # Add residual connection
        x_added = x + residual

        # Compute RMS normalization using rsqrt for efficiency (same as static version)
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        x_normalized = x_added * rstd
        
        # Apply scale parameter
        output = x_normalized * gamma

        # Cast to target data type
        output_cast = output.to(self.target_dtype)
        return output_cast

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    batch_size1, seq_len1, hidden_size1 = 16, 512, 2048
    x1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    residual1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    gamma1 = torch.randn(hidden_size1, dtype=torch.float32)

    # Case 2: Medium tensor size (24, 768, 2688) (non-aligned batch, medium hidden)
    batch_size2, seq_len2, hidden_size2 = 24, 768, 2688
    x2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    residual2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    gamma2 = torch.randn(hidden_size2, dtype=torch.float32)

    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    batch_size3, seq_len3, hidden_size3 = 32, 1024, 4096
    x3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    residual3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    gamma3 = torch.randn(hidden_size3, dtype=torch.float32)

    # Case 4: Very large tensor size (48, 1536, 8192) (non-aligned batch, larger than static)
    batch_size4, seq_len4, hidden_size4 = 48, 1536, 8192
    x4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    residual4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    gamma4 = torch.randn(hidden_size4, dtype=torch.float32)

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
    target_dtype = torch.float16
    return [epsilon, target_dtype]