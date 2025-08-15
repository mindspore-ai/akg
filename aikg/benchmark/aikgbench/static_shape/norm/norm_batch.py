import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Batch Normalization operation.
    Batch normalization normalizes the input across the batch dimension, which helps
    stabilize training and allows for higher learning rates. This operation computes
    batch statistics each time for benchmarking consistency.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, gamma, beta):
        """
        Perform Batch Normalization operation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            gamma: Scale parameter tensor of shape (hidden_size,) with float32 dtype
            beta: Shift parameter tensor of shape (hidden_size,) with float32 dtype

        Returns:
            Tuple of (output, mean, rstd) where output is the normalized tensor
        """
        # Compute batch statistics
        mean = x.mean(dim=(0, 1), keepdim=True)
        var = x.var(dim=(0, 1), keepdim=True, unbiased=False)

        # Compute reciprocal standard deviation
        rstd = 1.0 / torch.sqrt(var + self.epsilon)

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

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate scale and shift parameters
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)

    return [x, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Only epsilon is required
    return [1e-6]  # epsilon=1e-6