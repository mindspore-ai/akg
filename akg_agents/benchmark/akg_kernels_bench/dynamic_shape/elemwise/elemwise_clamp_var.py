import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, min_val=None, max_val=None):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input_tensor):
        # Clamp operation (also known as clip)
        # This operation is commonly used in neural networks for:
        # - Limiting the range of values in tensors
        # - Implementing gradient clipping during training
        # - Normalizing activations to prevent exploding gradients
        
        # Clamp the input tensor values to the specified range
        result = torch.clamp(input_tensor, min=self.min_val, max=self.max_val)
        
        return result

def get_inputs_dyn_list():
    # Clamp variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inputs1 = torch.randn(256, 1024, dtype=torch.float32) * 10
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inputs2 = torch.randn(1024, 4096, dtype=torch.float32) * 10
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inputs3 = torch.randn(2048, 8192, dtype=torch.float32) * 10
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs4 = torch.randn(125, 5120, dtype=torch.float32) * 10
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]

def get_init_inputs():
    # Fixed parameters for clamp operation
    min_val = -5.0  # Minimum value to clamp to
    max_val = 5.0   # Maximum value to clamp to
    
    return [min_val, max_val]