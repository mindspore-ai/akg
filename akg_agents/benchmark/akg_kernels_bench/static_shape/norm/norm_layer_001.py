import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Layer Normalization operation that normalizes the input tensor along the last dimension.
    Formula: output = (x - mean) / sqrt(variance + epsilon) * gamma + beta
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor, gamma, beta):
        # Calculate mean and variance along the last dimension
        mean = input_tensor.mean(dim=-1, keepdim=True)
        variance = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        normalized = (input_tensor - mean) / torch.sqrt(variance + self.epsilon)
        
        # Apply scale (gamma) and shift (beta) parameters
        result = normalized * gamma + beta
        return result


def get_inputs():
    # Batch size: 32
    # Hidden dimension: 4096
    # Sequence length: 512
    input_tensor = torch.randn(32, 512, 4096, dtype=torch.float32)
    gamma = torch.randn(4096, dtype=torch.float32)
    beta = torch.randn(4096, dtype=torch.float32)
    return [input_tensor, gamma, beta]


def get_init_inputs():
    # Parameters for Layer Normalization operation
    epsilon = 1e-5
    return [epsilon]