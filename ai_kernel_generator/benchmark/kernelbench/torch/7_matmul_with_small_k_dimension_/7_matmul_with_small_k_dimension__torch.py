import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)


M = 16384
N = 16384
K = 32


def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
