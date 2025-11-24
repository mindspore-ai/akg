import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Inplace Add + RMS Normalization fused operation.
    This operation adds a residual connection to the input in-place and then applies RMS normalization.
    RMS normalization is computationally more efficient than full layer normalization as it only
    normalizes by the RMS without subtracting the mean.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, residual, gamma):
        """
        Perform Inplace Add + RMS Normalization fused operation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            residual: Residual tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            gamma: Scale parameter tensor of shape (hidden_size,) with float32 dtype

        Returns:
            Output tensor after residual addition and RMS normalization
        """
        # Add residual connection in-place
        x.add_(residual)

        # Compute reciprocal RMS and normalize in fewer temporaries (same as static version)
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x * rstd) * gamma
        return output

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    batch_size1, seq_len1, hidden_size1 = 16, 512, 2048
    x1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    residual1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    gamma1 = torch.randn(hidden_size1, dtype=torch.float32)

    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    batch_size2, seq_len2, hidden_size2 = 24, 768, 3072
    x2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    residual2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    gamma2 = torch.randn(hidden_size2, dtype=torch.float32)

    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    batch_size3, seq_len3, hidden_size3 = 32, 1024, 4096
    x3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    residual3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    gamma3 = torch.randn(hidden_size3, dtype=torch.float32)

    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    batch_size4, seq_len4, hidden_size4 = 48, 1536, 6144
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
    return [epsilon]