import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs RMS Normalization Gradient operation.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, grad_output, x, gamma, rstd):
        """
        Perform RMS Normalization Gradient operation.
        """
        # Normalized input from forward pass
        x_normalized = x * rstd

        # Gradients for gamma and x
        grad_gamma = (grad_output * x_normalized).sum(dim=(0, 1))
        grad_x = (grad_output * gamma) * rstd
        return grad_x, grad_gamma


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    grad_output = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return [grad_output, x, gamma, rstd]


def get_init_inputs():
    return [1e-6]  # epsilon