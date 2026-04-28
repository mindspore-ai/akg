import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.sqrt(input, *, out=None)
        # Returns a new tensor with the square root of the elements of input.
        # This operation is commonly used in neural networks for:
        # - Normalization operations (e.g., RMS normalization)
        # - Computing standard deviations
        # - Mathematical transformations in specialized layers
        return torch.sqrt(input_tensor)


def get_inputs_dyn_list():
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    # Using positive values for sqrt since sqrt of negative numbers is not defined in reals
    inputs1 = torch.rand(256, 1024, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    # Using positive values for sqrt since sqrt of negative numbers is not defined in reals
    inputs2 = torch.rand(1024, 4096, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    # Using positive values for sqrt since sqrt of negative numbers is not defined in reals
    inputs3 = torch.rand(2048, 8192, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    # Using positive values for sqrt since sqrt of negative numbers is not defined in reals
    inputs4 = torch.rand(125, 5120, dtype=torch.float32) + 1e-6  # Adding small value to ensure positive values
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]


def get_init_inputs():
    # No parameters needed for sqrt
    # Extract params
    return []