import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, mat1, mat2):
        # Transpose inputs to match intended GEMM layout, then perform mm
        mat1 = mat1.t()
        mat2 = mat2.t()
        return torch.mm(mat1, mat2)


def get_inputs():
    # Using shapes that are representative of large model computations in fully connected layers
    # Shape (1344, 1024) and (4096, 1344) represent weight matrices that will be transposed
    mat1 = torch.randn(1344, 1024, dtype=torch.float16)  # Will be transposed to (1024, 1344)
    mat2 = torch.randn(4096, 1344, dtype=torch.float16)  # Will be transposed to (1344, 4096)
    return [mat1, mat2]


def get_init_inputs():
    # No parameters required
    return []