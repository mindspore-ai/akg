import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RMS Normalization (std path) that normalizes the input tensor using
    standard deviation instead of reciprocal standard deviation.
    Formula: output = (x / sqrt(mean(x^2) + epsilon)) * gamma
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma):
        # Compute standard deviation path
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.epsilon)
        result = input_tensor / std
        result = result * gamma
        return result

def get_inputs_dyn_list():
    # RMS normalization (std path) variation cases for diversity testing
    
    # Case 1: Standard tensor size (16, 512, 2048) (smaller than static)
    input_tensor1 = torch.randn(16, 512, 2048, dtype=torch.float32)
    gamma1 = torch.randn(2048, dtype=torch.float32)
    
    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    input_tensor2 = torch.randn(24, 768, 3072, dtype=torch.float32)
    gamma2 = torch.randn(3072, dtype=torch.float32)
    
    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    input_tensor3 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    gamma3 = torch.randn(4096, dtype=torch.float32)
    
    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    input_tensor4 = torch.randn(48, 1536, 6144, dtype=torch.float32)
    gamma4 = torch.randn(6144, dtype=torch.float32)
    
    return [
        [input_tensor1, gamma1],
        [input_tensor2, gamma2],
        [input_tensor3, gamma3],
        [input_tensor4, gamma4]
    ]

def get_init_inputs():
    epsilon = 1e-5
    return [epsilon]
