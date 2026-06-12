import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Symmetric square: A2 = A @ A where A is symmetric.
    A is [B, M, M] symmetric (from g @ g.mT), A2 = A @ A is [B, M, M] symmetric.
    Triton optimization: only store/read upper triangle of A, mirror for lower.
    Computes upper triangle + diagonal of A2, mirrors to lower triangle.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        squeeze_back = A.ndim == 2
        if squeeze_back:
            A = A.unsqueeze(0)
        A2 = A @ A
        if squeeze_back:
            A2 = A2.squeeze(0)
        return A2


B, M, N = 48, 1280, 3584


def get_inputs():
    g = torch.rand(B, M, N, dtype=torch.bfloat16) - 0.5
    A = g @ g.mT  # A is symmetric
    return [A]


def get_init_inputs():
    return []