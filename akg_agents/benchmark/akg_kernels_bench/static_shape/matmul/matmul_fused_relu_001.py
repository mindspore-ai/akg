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
    mat1 = torch.randn(1000, 8192, dtype=torch.float16)
    mat2 = torch.randn(8192, 8192, dtype=torch.float16)
    return [mat1, mat2]


def get_init_inputs():
    # No parameters required
    return []