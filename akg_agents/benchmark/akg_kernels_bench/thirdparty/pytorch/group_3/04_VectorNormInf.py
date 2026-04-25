import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the Infinity norm (max absolute value) of a vector
    along a specified dimension. Also known as the Chebyshev norm.
    Uses a max-reduction instead of sum-reduction, which is a fundamentally
    different compute pattern from L1/L2 norms.
    """
    def __init__(self):
        """
        Initializes the VectorNormInf layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Infinity norm (max absolute value) along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Infinity norm values of shape (batch_size,).
        """
        return torch.linalg.vector_norm(x, ord=float('inf'), dim=-1)

batch_size = 256
dim = 65536

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
