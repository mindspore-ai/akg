import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Exponential activation function operation.
    This operation is commonly used in neural networks for:
    - Activation function that computes element-wise exponential
    - Used in some probabilistic models and attention mechanisms
    - Maps input values to positive outputs
    
    Formula: output = exp(input)
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Exponential activation function applied to input_tensor
        result = torch.exp(input_tensor)
        return result

def get_inputs():
    # Batch size: 32
    # Sequence length: 512
    # Hidden size: 1024
    input_tensor = torch.randn(32, 512, 1024, dtype=torch.float32) * 0.1  # Scale down to prevent overflow
    return [input_tensor]

def get_init_inputs():
    # No parameters for Exponential activation operation
    return []