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
    # RMS normalization with small tensor variation cases for diversity testing
    
    # Case 1: Small tensor size (8, 256, 1344) (non-aligned hidden, smaller than static)
    input_tensor1 = torch.randn(8, 256, 1344, dtype=torch.float32)
    gamma1 = torch.randn(1344, dtype=torch.float32)
    
    # Case 2: Small tensor size (12, 384, 2016) (non-aligned batch, medium hidden)
    input_tensor2 = torch.randn(12, 384, 2016, dtype=torch.float32)
    gamma2 = torch.randn(2016, dtype=torch.float32)
    
    # Case 3: Small tensor size (16, 512, 2688) (aligned, same as static)
    input_tensor3 = torch.randn(16, 512, 2688, dtype=torch.float32)
    gamma3 = torch.randn(2688, dtype=torch.float32)
    
    # Case 4: Small tensor size (20, 640, 3360) (non-aligned, larger than static)
    input_tensor4 = torch.randn(20, 640, 3360, dtype=torch.float32)
    gamma4 = torch.randn(3360, dtype=torch.float32)
    
    return [
        [input_tensor1, gamma1],
        [input_tensor2, gamma2],
        [input_tensor3, gamma3],
        [input_tensor4, gamma4]
    ]

def get_init_inputs():
    epsilon = 1e-5
    return [epsilon]