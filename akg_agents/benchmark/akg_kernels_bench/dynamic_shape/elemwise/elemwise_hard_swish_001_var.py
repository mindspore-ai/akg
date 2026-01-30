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
    # Case 1: Small (batch=16, seq=64, hidden=512)
    var1 = torch.randn(16, 64, 512, dtype=torch.float32)

    # Case 2: Middle (batch=32, seq=512, hidden=1024)
    var2 = torch.randn(32, 512, 1024, dtype=torch.float32)

    # Case 3: Large (batch=256, seq=1024, hidden=4096)
    var3 = torch.randn(256, 1024, 4096, dtype=torch.float32)

    # Case 4: Non-aligned (batch=48, seq=256, hidden=2688)
    var4 = torch.randn(48, 256, 2688, dtype=torch.float32)
    
    return [
        [var1],
        [var2],
        [var3],
        [var4]
    ]

def get_init_inputs():
    # No parameters for Hard Swish activation operation
    return []