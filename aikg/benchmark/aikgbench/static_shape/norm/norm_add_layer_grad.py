import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Add + Layer Normalization Gradient operation.
    This operation computes gradients for the add + layer normalization fusion operation.
    It's commonly used in training neural networks where the forward pass includes
    residual connections followed by layer normalization.
    """

    def __init__(self, epsilon=1e-6):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, grad_output, x, residual, gamma, beta, mean, rstd):
        """
        Perform Add + Layer Normalization Gradient operation.

        Args:
            grad_output: Gradient of the output of shape (batch_size, seq_len, hidden_size) with float32 dtype
            x: Input tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            residual: Residual tensor of shape (batch_size, seq_len, hidden_size) with float32 dtype
            gamma: Scale parameter tensor of shape (hidden_size,) with float32 dtype
            beta: Shift parameter tensor of shape (hidden_size,) with float32 dtype
            mean: Mean tensor from forward pass of shape (batch_size, seq_len, 1) with float32 dtype
            rstd: Reciprocal standard deviation from forward pass of shape (batch_size, seq_len, 1) with float32 dtype

        Returns:
            Tuple of (grad_x, grad_residual, grad_gamma, grad_beta) gradients
        """
        # Compute the normalized input from forward pass
        x_added = x + residual
        x_normalized = (x_added - mean) * rstd

        # Gradient with respect to beta
        grad_beta = grad_output.sum(dim=(0, 1))

        # Gradient with respect to gamma
        grad_gamma = (grad_output * x_normalized).sum(dim=(0, 1))

        # Gradient with respect to normalized input
        grad_normalized = grad_output * gamma

        # Gradients w.r.t. x and residual are identical to grad_normalized
        return grad_normalized, grad_normalized, grad_gamma, grad_beta


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

    # Generate gradient output
    grad_output = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate mean and rstd from forward pass
    x_added = x + residual
    mean = x_added.mean(dim=-1, keepdim=True)
    var = x_added.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + 1e-6)

    return [grad_output, x, residual, gamma, beta, mean, rstd]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    # Return parameters as a list
    epsilon = 1e-6
    return [epsilon]