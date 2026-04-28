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


def get_inputs():
    # Batch size: 32
    # Feature channels: 4096
    # Spatial dimensions: 8 x 8
    input_tensor = torch.randn(32, 4096, 8, 8, dtype=torch.float32)
    return [input_tensor]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    epsilon = 1e-5
    return [epsilon]