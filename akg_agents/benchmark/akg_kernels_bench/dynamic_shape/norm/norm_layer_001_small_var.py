import torch
import torch.nn as nn

class Model(nn.Module):
    # Formula: output = (x - mean) / sqrt(variance + epsilon) * gamma + beta

    def __init__(self, normalized_shape):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, input_tensor):
        # torch.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05, cudnn_enable=True)
        # Applies Layer Normalization over a mini-batch of inputs.
        
        # Create weight and bias tensors with the correct shape
        weight = torch.ones(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        bias = torch.zeros(self.normalized_shape, dtype=input_tensor.dtype, device=input_tensor.device)
        
        return torch.layer_norm(input_tensor, self.normalized_shape, weight, bias)

def get_inputs_dyn_list():
    # Layer normalization with small tensors variation cases with both aligned and non-aligned shapes
    # All tensors use 256 as the last dimension to match normalized_shape
    
    # Case 1: Small tensor size (2, 15, 256) (non-aligned batch and sequence)
    input_tensor1 = torch.randn(2, 15, 256, dtype=torch.float32)
    
    # Case 2: Small tensor size (4, 32, 256) (aligned)
    input_tensor2 = torch.randn(4, 32, 256, dtype=torch.float32)
    
    # Case 3: Medium tensor size (8, 16, 256) (aligned)
    input_tensor3 = torch.randn(8, 16, 256, dtype=torch.float32)
    
    # Case 4: Medium tensor size (16, 255, 256) (non-aligned sequence)
    input_tensor4 = torch.randn(16, 255, 256, dtype=torch.float32)
    
    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4]
    ]

def get_init_inputs():
    # Parameters for layer_norm
    # The normalized_shape should match the last dimension of input tensors
    normalized_shape = (256,)
    return [normalized_shape]