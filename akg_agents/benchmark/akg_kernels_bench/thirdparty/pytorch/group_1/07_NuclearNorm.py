import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the nuclear norm (trace norm) of a matrix.
    The nuclear norm is defined as the sum of singular values of the matrix.
    Involves SVD decomposition, making it computationally expensive.
    """
    def __init__(self):
        """
        Initializes the NuclearNorm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the nuclear norm of each matrix in a batch.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols).

        Returns:
            torch.Tensor: Nuclear norm values of shape (batch_size,).
        """
        return torch.linalg.matrix_norm(x, ord='nuc')

batch_size = 16
rows = 512
cols = 512

def get_inputs():
    x = torch.randn(batch_size, rows, cols)
    return [x]

def get_init_inputs():
    return []
