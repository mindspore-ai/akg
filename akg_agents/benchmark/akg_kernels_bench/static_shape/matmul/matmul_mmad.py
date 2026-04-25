import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, matrix1, matrix2):
        # Custom matrix multiplication operation (Mmad)
        # Performs matrix multiplication of the matrices matrix1 and matrix2.
        # Matrix multiplication is a fundamental operation in neural networks used for:
        # - Linear transformations in fully connected layers
        # - Attention computations in transformers
        # - Convolution operations when expressed as matrix multiplication
        return torch.matmul(matrix1, matrix2)


def get_inputs():
    # Using shapes that are representative of large model computations
    # Shape (1024, 1344) and (1344, 4096) represent typical matrix multiplication dimensions
    # in transformer models (e.g., projecting from embedding dimension to hidden dimension)
    matrix1 = torch.randn(1024, 1344, dtype=torch.float32)
    matrix2 = torch.randn(1344, 4096, dtype=torch.float32)
    return [matrix1, matrix2]


def get_init_inputs():
    # No parameters needed for mmad
    return []