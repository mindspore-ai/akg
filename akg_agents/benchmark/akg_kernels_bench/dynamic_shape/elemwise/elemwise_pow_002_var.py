import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, exponent):
        # torch.pow(input, exponent, *, out=None)
        # Takes the power of each element in input with exponent and returns a tensor with the result.
        # This is a power operation with exponent=0.5 (square root).
        # Power operations are commonly used in neural networks for:
        # - Implementing polynomial activation functions
        # - Computing distance metrics
        # - Mathematical transformations in specialized layers
        return torch.pow(input_tensor, exponent)


def get_inputs_dyn_list():
    # Element-wise power (square root) variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inp1_1 = torch.rand(256, 4096, dtype=torch.float32) + 1e-6  # Using positive values for square root
    inp2_1 = torch.full((256, 4096), 0.5, dtype=torch.float32)  # Square root operation
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inp1_2 = torch.rand(125, 5120, dtype=torch.float32) + 1e-6  # Using positive values for square root
    inp2_2 = torch.full((125, 5120), 0.5, dtype=torch.float32)  # Square root operation
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inp1_3 = torch.rand(512, 6144, dtype=torch.float32) + 1e-6  # Using positive values for square root
    inp2_3 = torch.full((512, 6144), 0.5, dtype=torch.float32)  # Square root operation
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inp1_4 = torch.rand(1024, 8192, dtype=torch.float32) + 1e-6  # Using positive values for square root
    inp2_4 = torch.full((1024, 8192), 0.5, dtype=torch.float32)  # Square root operation
    
    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4]
    ]


def get_init_inputs():
    # No parameters needed for pow
    return []