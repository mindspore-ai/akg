import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that computes the matrix infinity norm (max absolute row sum)
    for a batch of matrices.
    Matrix infinity norm: max_i(sum_j(|a_ij|)), i.e., the maximum L1 norm
    among all rows of the matrix.
    """
    def __init__(self):
        """
        Initializes the MatrixInfNorm layer.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the matrix infinity norm for each matrix in the batch.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, rows, cols).

        Returns:
            torch.Tensor: Matrix infinity norm values of shape (batch_size,).
        """
        return torch.linalg.matrix_norm(x, ord=float('inf'))

batch_size = 16
rows = 2048
cols = 2048

def get_inputs():
    x = torch.randn(batch_size, rows, cols)
    return [x]

def get_init_inputs():
    return []
