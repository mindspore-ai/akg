import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Instance Normalization operation.
    Instance normalization normalizes across spatial dimensions (H, W) 
    for each sample and each channel independently.
    """
    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_tensor):
        """
        Perform Instance Normalization operation.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Instance normalized output tensor
        """
        # Apply instance normalization
        result = torch.nn.functional.instance_norm(input_tensor, eps=self.epsilon)
        
        return result

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 2048, 4, 4) (smaller than static)
    input_tensor1 = torch.randn(16, 2048, 4, 4, dtype=torch.float32)
    
    # Case 2: Medium tensor size (24, 3072, 6, 6) (non-aligned batch, medium channels)
    input_tensor2 = torch.randn(24, 3072, 6, 6, dtype=torch.float32)
    
    # Case 3: Large tensor size (32, 4096, 8, 8) (aligned, same as static)
    input_tensor3 = torch.randn(32, 4096, 8, 8, dtype=torch.float32)
    
    # Case 4: Very large tensor size (48, 6144, 12, 12) (non-aligned batch, larger than static)
    input_tensor4 = torch.randn(48, 6144, 12, 12, dtype=torch.float32)
    
    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    epsilon = 1e-5
    return [epsilon]