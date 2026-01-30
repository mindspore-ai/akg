import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(Model, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input_tensor, mat1, mat2, bias):
        # Compute: alpha * (mat1 @ mat2) + beta * input_tensor using torch.mm
        mm_result = torch.mm(mat1, mat2)
        result = self.alpha * mm_result + self.beta * input_tensor
        result = result + bias
        
        return result


def get_inputs():
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    mat1 = torch.randn(1024, 1344, dtype=torch.float32)
    mat2 = torch.randn(1344, 4096, dtype=torch.float32)
    bias = torch.randn(4096, dtype=torch.float32)  # 1D bias term
    return [input_tensor, mat1, mat2, bias]


def get_init_inputs():
    alpha = 1.0
    beta = 1.0
    return [alpha, beta]