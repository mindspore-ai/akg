import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the L2 (Euclidean) vector norm along a specified dimension.
    Uses torch.linalg.vector_norm for efficient norm computation.
    """
    def __init__(self):
        """
        Initializes the VectorNorm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the L2 vector norm along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Norm values of shape (batch_size,).
        """
        return torch.linalg.vector_norm(x, ord=2, dim=-1)

batch_size = 128
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
