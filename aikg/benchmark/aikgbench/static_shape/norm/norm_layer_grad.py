import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Layer Normalization Gradient operation.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, grad_output, x, gamma, beta, mean, rstd):
        """
        Perform Layer Normalization Gradient operation.
        """
        # Normalize using saved mean and rstd
        x_normalized = (x - mean) * rstd

        # Gradients for beta and gamma
        grad_beta = grad_output.sum(dim=(0, 1))
        grad_gamma = (grad_output * x_normalized).sum(dim=(0, 1))

        # Gradient wrt input x (simplified)
        grad_x = (grad_output * gamma) * rstd
        return grad_x, grad_gamma, grad_beta


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    gamma = torch.randn(hidden_size, dtype=torch.float32)
    beta = torch.randn(hidden_size, dtype=torch.float32)
    grad_output = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + 1e-6)
    return [grad_output, x, gamma, beta, mean, rstd]


def get_init_inputs():
    epsilon = 1e-6
    return [epsilon]