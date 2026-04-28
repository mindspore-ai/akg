import torch
import torch.nn as nn

class Model(nn.Module):
    """
    RMS Normalization operation that normalizes the input tensor using RMS normalization.
    This operation is commonly used in neural networks for:
    - Normalizing activations in transformer models
    - Used in LLaMA and other large language models
    - Provides an alternative to Layer Normalization with better performance
    
    Formula: output = (x / sqrt(mean(x^2) + epsilon)) * gamma
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma):
        # RMS normalization with gamma parameter
        # Calculate RMS normalization
        variance = input_tensor.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        result = input_tensor * rstd
        
        # Apply scale parameter (gamma)
        result = result * gamma
        return result

def get_inputs_dyn_list():
    # RMS normalization with large tensors variation cases for diversity testing
    
    # Case 1: Large tensor size (64, 512, 4096) (smaller than static, non-aligned batch)
    input_tensor1 = torch.randn(64, 512, 4096, dtype=torch.float32)
    gamma1 = torch.randn(4096, dtype=torch.float32)
    
    # Case 2: Large tensor size (96, 768, 6144) (non-aligned batch, medium hidden)
    input_tensor2 = torch.randn(96, 768, 6144, dtype=torch.float32)
    gamma2 = torch.randn(6144, dtype=torch.float32)
    
    # Case 3: Large tensor size (128, 1024, 8192) (aligned, same as static)
    input_tensor3 = torch.randn(128, 1024, 8192, dtype=torch.float32)
    gamma3 = torch.randn(8192, dtype=torch.float32)
    
    # Case 4: Ultra large tensor size (160, 1280, 10240) (non-aligned batch, larger than static)
    input_tensor4 = torch.randn(160, 1280, 10240, dtype=torch.float32)
    gamma4 = torch.randn(10240, dtype=torch.float32)
    
    return [
        [input_tensor1, gamma1],
        [input_tensor2, gamma2],
        [input_tensor3, gamma3],
        [input_tensor4, gamma4]
    ]

def get_init_inputs():
    epsilon = 1e-5
    return [epsilon]