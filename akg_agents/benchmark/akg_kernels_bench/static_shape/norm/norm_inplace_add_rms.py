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
        """
        # Add residual connection in-place
        x.add_(residual)

        # Compute reciprocal RMS and normalize in fewer temporaries
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        output = (x * rstd) * gamma
        return output


def get_inputs():
    # Batch size: 32
    # Sequence length: 1024
    # Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)

    return [x, residual, gamma]


def get_init_inputs():
    # epsilon only
    return [1e-6]