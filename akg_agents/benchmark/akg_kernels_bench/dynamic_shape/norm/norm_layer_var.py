import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs Layer Normalization operation with dynamic shapes.
    Layer normalization normalizes across the feature dimension for each sample.
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        """
        Perform Layer Normalization operation.
        
        Args:
            input_tensor: Input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Layer normalized output tensor
        """
        # For dynamic shape cases, we determine the normalized_shape from the input tensor
        normalized_shape = (input_tensor.shape[-1],)  # Normalizing over the last dimension
        
        # Create weight and bias tensors with the correct shape
        weight = torch.ones(normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        bias = torch.zeros(normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        return torch.layer_norm(input_tensor, normalized_shape, weight, bias)


def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    input_tensor1 = torch.randn(16, 512, 2048, dtype=torch.float32)
    
    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    input_tensor2 = torch.randn(24, 768, 3072, dtype=torch.float32)
    
    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    input_tensor3 = torch.randn(32, 1024, 4096, dtype=torch.float32)
    
    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    input_tensor4 = torch.randn(48, 1536, 6144, dtype=torch.float32)
    
    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    For dynamic shape cases, no initialization parameters needed.
    """
    return []