import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape=4096, eps=1e-6):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, tensor1, tensor2, weight, bias):
        # AddRmsNorm operation
        # This operation is commonly used in neural networks for:
        # - Combining residual connections with RMS normalization
        # - Used in transformer architectures like T5
        # - Providing an alternative to LayerNorm with potentially better performance
        
        # Perform addition
        added = tensor1 + tensor2
        
        # Apply RMS normalization
        rms = torch.sqrt(torch.mean(added ** 2, dim=-1, keepdim=True) + self.eps)
        result = added / rms * weight + bias
        
        return result

def get_inputs():
    tensor1 = torch.randn(2048, 4096, dtype=torch.float32)
    tensor2 = torch.randn(2048, 4096, dtype=torch.float32)
    weight = torch.ones(4096, dtype=torch.float32)
    bias = torch.zeros(4096, dtype=torch.float32)
    return [tensor1, tensor2, weight, bias]

def get_init_inputs():
    normalized_shape = 4096
    eps = 1e-6
    return [normalized_shape, eps]