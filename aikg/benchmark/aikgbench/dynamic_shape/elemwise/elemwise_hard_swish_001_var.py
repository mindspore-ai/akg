import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hard Swish activation function operation.
    This operation is commonly used in neural networks for:
    - Efficient activation function for mobile and edge devices
    - Used in MobileNetV3 and other efficient architectures
    - Provides a piecewise linear approximation to Swish
    
    Formula: output = input * relu6(input + 3) / 6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Hard Swish activation function applied to input_tensor
        result = torch.nn.functional.hardswish(input_tensor)
        return result

def get_inputs_dyn_list():
    # Hard Swish activation variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small shapes with aligned dimensions
    # Shape (32, 512, 1024) represents a batch of 32 samples, sequence length 512, hidden size 1024
    var1 = torch.randn(32, 512, 1024, dtype=torch.float32)
    
    # Case 2: Medium shapes with aligned dimensions
    # Shape (256, 1024, 4096) represents a batch of 256 samples, sequence length 1024, hidden size 4096
    var2 = torch.randn(256, 1024, 4096, dtype=torch.float32)
    
    # Case 3: Large shapes with aligned dimensions
    # Shape (1024, 2048, 8192) represents a batch of 1024 samples, sequence length 2048, hidden size 8192
    var3 = torch.randn(1024, 2048, 8192, dtype=torch.float32)
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 256, 5120) represents a batch of 125 samples, sequence length 256, hidden size 5120
    var4 = torch.randn(125, 256, 5120, dtype=torch.float32)
    
    return [
        [var1],
        [var2],
        [var3],
        [var4]
    ]

def get_init_inputs():
    # No parameters for Hard Swish activation operation
    return []