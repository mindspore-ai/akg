import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs matrix-vector multiplication (C = A * B).
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix-vector multiplication.

        Args:
            A: Input matrix of shape (M, K).
            B: Input vector of shape (K, 1).

        Returns:
            Output vector of shape (M, 1).
        """
        return torch.matmul(A, B)


M = 256
K = 131072


def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed
