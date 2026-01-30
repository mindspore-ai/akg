import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Swish (SiLU) activation function operation.
    This operation is commonly used in neural networks for:
    - Activation function in EfficientNet and other modern architectures
    - Used in various transformer models
    - Provides smooth, non-monotonic activation
    
    Formula: output = x * sigmoid(x * scale)
    """
    def __init__(self, scale=1.0):
        super(Model, self).__init__()
        self.scale = scale

    def forward(self, input_tensor):
        # Swish activation function: input_tensor * sigmoid(input_tensor * scale)
        result = input_tensor * torch.sigmoid(input_tensor * self.scale)
        return result

def get_inputs_dyn_list():
    # Swish activation variation cases with both aligned and non-aligned shapes
    
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
    # Fixed parameters for Swish activation operation
    scale = 1.0  # Scaling factor for the activation
    return [scale]