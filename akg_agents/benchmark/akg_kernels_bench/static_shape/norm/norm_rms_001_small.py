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


def get_inputs():
    # Batch size: 16
    # Hidden dimension: 2688
    # Sequence length: 512
    input_tensor = torch.randn(16, 512, 2688, dtype=torch.float32)
    gamma = torch.randn(2688, dtype=torch.float32)
    return [input_tensor, gamma]


def get_init_inputs():
    # Parameters for RMS Normalization operation
    epsilon = 1e-5
    return [epsilon]