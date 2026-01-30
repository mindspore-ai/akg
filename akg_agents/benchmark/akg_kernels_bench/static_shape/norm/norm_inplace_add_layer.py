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

def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32
    # Sequence length: 1024
    # Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096

    # Generate input and residual tensors
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate scale and shift parameters
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)

    return [x, residual, gamma, beta]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    return [1e-6]  # epsilon=1e-6