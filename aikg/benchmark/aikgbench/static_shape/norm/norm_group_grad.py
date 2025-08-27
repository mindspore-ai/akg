import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Group Normalization Gradient operation.
    """

    def __init__(self, num_groups=32, epsilon=1e-6):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.epsilon = epsilon

    def forward(self, grad_output, x, gamma, beta, mean, rstd):
        """
        Perform Group Normalization Gradient operation.
        """
        batch_size, seq_len, hidden_size = x.shape
        groups = self.num_groups
        channels_per_group = hidden_size // groups

        # Reshape for group normalization computation
        x_reshaped = x.view(batch_size, seq_len, groups, channels_per_group)

        # Normalized input from forward pass
        x_normalized = (x_reshaped - mean) * rstd

        # Gradients for beta and gamma
        grad_beta = grad_output.sum(dim=(0, 1))
        grad_gamma = (grad_output * x_normalized.view(batch_size, seq_len, hidden_size)).sum(dim=(0, 1))

        # Gradient wrt input x (simplified)
        grad_x = (grad_output * gamma).view(batch_size, seq_len, groups, channels_per_group) * rstd
        grad_x = grad_x.view(batch_size, seq_len, hidden_size)
        return grad_x, grad_gamma, grad_beta


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    num_groups = 32
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)
    grad_output = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    x_reshaped = x.view(batch_size, seq_len, num_groups, hidden_size // num_groups)
    mean = x_reshaped.mean(dim=-1, keepdim=True)
    var = x_reshaped.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + 1e-6)
    return [grad_output, x, gamma, beta, mean, rstd]


def get_init_inputs():
    return [32, 1e-6]  # num_groups, epsilon