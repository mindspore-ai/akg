import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Matrix multiplication operation that performs linear transformation.
    This operation is commonly used in neural networks for:
    - Fully connected layers in neural networks
    - Projection layers in transformers
    - Used in virtually all neural network architectures
    Formula: output = input @ weight^T + bias
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, weight, bias):
        result = torch.matmul(input_tensor, weight)
        result = result + bias
        return result

def get_inputs():
    input_tensor = torch.randn(640, 8192, dtype=torch.float16)
    weight = torch.randn(8192, 4096, dtype=torch.float16)
    bias = torch.randn(4096, dtype=torch.float16)
    return [input_tensor, weight, bias]

def get_init_inputs():
    return []