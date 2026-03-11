import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the L1 vector norm (Manhattan norm / sum of absolute values)
    along a specified dimension.
    Unlike L1 normalization (F.normalize p=1), this returns the norm value itself,
    not the normalized tensor.
    """
    def __init__(self):
        """
        Initializes the VectorNormL1 layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the L1 vector norm along the last dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: L1 norm values of shape (batch_size,).
        """
        return torch.linalg.vector_norm(x, ord=1, dim=-1)

batch_size = 128
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
