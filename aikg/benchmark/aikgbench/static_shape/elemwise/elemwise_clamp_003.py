import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Element-wise clamp operation (2D, FP32).
    Medium scale: e7
    """
    def __init__(self, min_val=None, max_val=None):
        super(Model, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input_tensor):
        # 2D clamp operation
        result = torch.clamp(input_tensor, min=self.min_val, max=self.max_val)
        
        return result

def get_inputs():
    # Large scale: 4096 * 4096 ≈ e7

    input_tensor = torch.randn(4096, 4096, dtype=torch.float32) * 10  # Multiply by 10 to get a wider range of values
    return [input_tensor]

def get_init_inputs():
    # Parameters for clamp operation
    min_val = -5.0  # Minimum value to clamp to
    max_val = 5.0   # Maximum value to clamp to
    
    return [min_val, max_val]