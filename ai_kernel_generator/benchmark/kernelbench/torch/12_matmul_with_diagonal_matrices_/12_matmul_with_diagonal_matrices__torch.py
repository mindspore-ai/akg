import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        return torch.diag(A) @ B


M = 4096
N = 4096


def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
