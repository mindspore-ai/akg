import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, var):
        # torch.max(input, dim, keepdim=False)
        # Returns a namedtuple (values, indices) where values is the maximum value of each row
        # of the input tensor in the given dimension dim, and indices is the index location of
        # each maximum value found.
        # This is a max operation along dimension 0.
        # Max operations are commonly used in neural networks for:
        # - Max pooling in convolutional networks
        # - Finding the most activated neuron in a layer
        # - Attention mechanisms in transformers
        return torch.max(var, self.dim)


def get_inputs_dyn_list():
    # Max reduction along dimension 0 variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small tensor size 15x15 (non-aligned)
    inputs1 = torch.randn(15, 15, dtype=torch.float32)
    
    # Case 2: Small tensor size 31x31 (non-aligned)
    inputs2 = torch.randn(31, 31, dtype=torch.float32)
    
    # Case 3: Small tensor size 32x32 (aligned)
    inputs3 = torch.randn(32, 32, dtype=torch.float32)
    
    # Case 4: Medium tensor size 127x127 (non-aligned)
    inputs4 = torch.randn(127, 127, dtype=torch.float32)
    
    # Case 5: Medium tensor size 128x128 (aligned)
    inputs5 = torch.randn(128, 128, dtype=torch.float32)
    
    # Case 6: Large tensor size 511x511 (non-aligned)
    inputs6 = torch.randn(511, 511, dtype=torch.float32)
    
    # Case 7: Large tensor size 512x512 (aligned)
    inputs7 = torch.randn(512, 512, dtype=torch.float32)
    
    # Case 8: Very large tensor size 1023x1023 (non-aligned)
    inputs8 = torch.randn(1023, 1023, dtype=torch.float32)
    
    # Case 9: Very large tensor size 1024x1024 (aligned)
    inputs9 = torch.randn(1024, 1024, dtype=torch.float32)
    
    # Case 10: Extreme tensor size 4095x4095 (non-aligned)
    inputs10 = torch.randn(4095, 4095, dtype=torch.float32)
    
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4],
        [inputs5],
        [inputs6],
        [inputs7],
        [inputs8],
        [inputs9],
        [inputs10]
    ]


def get_init_inputs():
    # Fixed parameters for max reduction along dimension 0
    dim = 0  # Reduce along first dimension (batch dimension)
    return [dim]
