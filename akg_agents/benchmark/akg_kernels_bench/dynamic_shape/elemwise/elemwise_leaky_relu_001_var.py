import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Leaky ReLU activation function operation.
    This operation is commonly used in neural networks for:
    - Activation function that addresses "dying ReLU" problem
    - Allows small negative values when input is negative
    - Used in some deep learning architectures
    
    Formula: output = max(input, negative_slope * input)
    """
    def __init__(self, negative_slope=0.01):
        super(Model, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, input_tensor):
        # Leaky ReLU activation function applied to input_tensor
        result = torch.nn.functional.leaky_relu(input_tensor, negative_slope=self.negative_slope)
        return result

def get_inputs_dyn_list():
    # Leaky ReLU activation variation cases with both aligned and non-aligned shapes
    
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
    # Fixed parameters for Leaky ReLU activation operation
    negative_slope = 0.01  # Controls the angle of negative slope
    return [negative_slope]