import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, tensor1, tensor2):
        # torch.dot(input, other)
        # Computes the dot product of two 1D tensors.
        # The dot product is calculated as the sum of the element-wise products.
        # This operation is commonly used in neural networks for:
        # - Computing similarity between vectors
        # - Implementing attention mechanisms
        # - Calculating projections in certain layers
        return torch.dot(tensor1, tensor2)


def get_inputs():
    # Hidden dimension: 4096
    tensor1 = torch.randn(4096, dtype=torch.float32)
    tensor2 = torch.randn(4096, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    # No parameters needed for dot product
    return []