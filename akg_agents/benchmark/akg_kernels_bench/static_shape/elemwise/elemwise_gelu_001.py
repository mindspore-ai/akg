import torch
import torch.nn as nn

class Model(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) activation function operation.
    This operation is commonly used in neural networks for:
    - Activation function in transformer models (BERT, GPT)
    - Provides smooth approximation to ReLU
    - Often performs better than ReLU in deep networks
    
    Formula: output = 0.5 * input * (1 + erf(input / sqrt(2)))
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # GELU activation function applied to input_tensor
        result = torch.nn.functional.gelu(input_tensor)
        return result

def get_inputs():
    # Batch size: 32
    # Sequence length: 512
    # Hidden size: 1024
    input_tensor = torch.randn(32, 512, 1024, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    # No parameters for GELU activation operation
    return []