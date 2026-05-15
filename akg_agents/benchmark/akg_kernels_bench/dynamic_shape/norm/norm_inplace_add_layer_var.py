import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Inplace Add + Layer Normalization fused operation.
    This operation adds a residual connection to the input in-place and then applies layer normalization.
    The inplace operation saves memory by modifying the input tensor directly, which is useful
    in memory-constrained environments like large language model inference.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma, beta):
        """
        Perform Inplace Add + Layer Normalization fused operation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            residual: Residual tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            gamma: Scale parameter tensor of shape (hidden_size,) with float32 dtype
            beta: Shift parameter tensor of shape (hidden_size,) with float32 dtype

        Returns:
            Tuple of (output, mean, rstd) where output is the normalized tensor
        """
        # Add residual connection in-place
        x.add_(residual)

        # Compute mean and variance across the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + self.epsilon)

        # Apply normalization
        x_normalized = (x - mean) * rstd

        # Apply scale and shift parameters
        output = x_normalized * gamma + beta

        return output, mean, rstd

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    batch_size1, seq_len1, hidden_size1 = 16, 512, 2048
    x1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    residual1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    gamma1 = torch.randn(hidden_size1, dtype=torch.float32)
    beta1 = torch.randn(hidden_size1, dtype=torch.float32)

    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    batch_size2, seq_len2, hidden_size2 = 24, 768, 3072
    x2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    residual2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    gamma2 = torch.randn(hidden_size2, dtype=torch.float32)
    beta2 = torch.randn(hidden_size2, dtype=torch.float32)

    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    batch_size3, seq_len3, hidden_size3 = 32, 1024, 4096
    x3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    residual3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    gamma3 = torch.randn(hidden_size3, dtype=torch.float32)
    beta3 = torch.randn(hidden_size3, dtype=torch.float32)

    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    batch_size4, seq_len4, hidden_size4 = 48, 1536, 6144
    x4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    residual4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    gamma4 = torch.randn(hidden_size4, dtype=torch.float32)
    beta4 = torch.randn(hidden_size4, dtype=torch.float32)

    return [
        [x1, residual1, gamma1, beta1],
        [x2, residual2, gamma2, beta2],
        [x3, residual3, gamma3, beta3],
        [x4, residual4, gamma4, beta4]
    ]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]