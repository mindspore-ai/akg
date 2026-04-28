import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, mat1, mat2):
        # Perform mm then apply ReLU activation
        result = torch.mm(mat1, mat2)
        result = torch.relu(result)
        return result


def get_inputs():
    # Using shapes that are representative of large model computations in fully connected layers
    # Shape (1024, 1344) and (1344, 4096) represent weight matrices
    mat1 = torch.randn(1024, 1344, dtype=torch.float16)
    mat2 = torch.randn(1344, 4096, dtype=torch.float16)
    return [mat1, mat2]


def get_init_inputs():
    # No parameters required
    return []