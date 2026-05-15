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

def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 1024
    input_tensor = torch.randn(1024, 1024, dtype=torch.float32) * 10  # Multiply by 10 to get a wider range of values
    return [input_tensor]

def get_init_inputs():
    # Parameters for clamp operation
    min_val = -5.0  # Minimum value to clamp to
    max_val = 5.0   # Maximum value to clamp to
    
    return [min_val, max_val]