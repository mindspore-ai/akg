import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + RMS Normalization + Cast fused operation.
    This operation adds a residual connection to the input, applies RMS normalization,
    and then casts the result to a different data type. This fusion is commonly used
    in mixed-precision training and inference scenarios.
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

        # Compute RMS normalization using rsqrt for efficiency
        variance = x_added.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        x_normalized = x_added * rstd
        
        # Apply scale parameter
        output = x_normalized * gamma

        # Cast to target data type
        output_cast = output.to(self.target_dtype)
        return output_cast


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

    # Generate scale parameter
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [x, residual, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    return [1e-6, torch.float16]  # epsilon=1e-6, target_dtype=float16