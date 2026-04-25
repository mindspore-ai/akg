import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, matrix1, matrix2):
        # Transpose second input (B) only, then perform mm
        mat2 = matrix2.t()
        return torch.mm(matrix1, mat2)


def get_inputs():
    # Using shapes that are representative of large model computations
    # Only transpose the second matrix (B)
    # matrix1: shape (M, K), matrix2: shape (N, K) -> mm(matrix1, matrix2.t()) gives (M, N)
    matrix1 = torch.randn(1344, 1024, dtype=torch.float16)  # Not transposed (M=1344, K=1024)
    matrix2 = torch.randn(4096, 1024, dtype=torch.float16)  # Will be transposed to (K=1024, N=4096)
    return [matrix1, matrix2]


def get_init_inputs():
    # No parameters required
    return []