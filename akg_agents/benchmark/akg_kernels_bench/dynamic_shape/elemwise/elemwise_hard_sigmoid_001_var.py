import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Hard Sigmoid activation function operation.
    This operation is commonly used in neural networks for:
    - Efficient activation function for mobile and edge devices
    - Used in MobileNetV3 and other efficient architectures
    - Provides a piecewise linear approximation to Sigmoid
    
    Formula: output = relu6(input + 3) / 6
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # Hard Sigmoid activation function applied to input_tensor
        result = torch.nn.functional.hardsigmoid(input_tensor)
        return result

def get_inputs_dyn_list():
    # Hard Sigmoid activation variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    var1 = torch.randn(256, 4096, dtype=torch.float32)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    var2 = torch.randn(125, 5120, dtype=torch.float32)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    var3 = torch.randn(512, 6144, dtype=torch.float32)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    var4 = torch.randn(1024, 8192, dtype=torch.float32)
    
    return [
        [var1],
        [var2],
        [var3],
        [var4]
    ]

def get_init_inputs():
    # No parameters for Hard Sigmoid activation operation
    return []