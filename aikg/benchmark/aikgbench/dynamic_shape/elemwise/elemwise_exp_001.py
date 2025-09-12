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


def get_inputs_dyn_list():
    # Case 1: Small (batch=16, seq=64, hidden=512)
    input_tensor1 = torch.randn(16, 64, 512, dtype=torch.float32) * 0.1

    # Case 2: Middle (batch=32, seq=512, hidden=1024)
    input_tensor2 = torch.randn(32, 512, 1024, dtype=torch.float32) * 0.1

    # Case 3: Large (batch=64, seq=2048, hidden=4096)
    input_tensor3 = torch.randn(64, 2048, 4096, dtype=torch.float32) * 0.1

    # Case 4: Non-aligned (batch=48, seq=256, hidden=2688)
    input_tensor4 = torch.randn(48, 256, 2688, dtype=torch.float32) * 0.1

    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]

def get_init_inputs():
    # No parameters for Exponential activation operation
    return []