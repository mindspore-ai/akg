import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.log(input, *, out=None)
        # Computes the element-wise natural logarithm of the input tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing logarithmic functions in probability computations
        # - Loss functions (e.g., negative log-likelihood)
        # - Mathematical transformations
        return torch.log(input_tensor)


def get_inputs_dyn_list():
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    # Using positive values since log is undefined for negative values
    inputs1 = torch.rand(256, 4096, dtype=torch.float32) + 0.1  # Adding 0.1 to ensure positive values
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    # Using positive values since log is undefined for negative values
    inputs2 = torch.rand(125, 5120, dtype=torch.float32) + 0.1  # Adding 0.1 to ensure positive values
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    # Using positive values since log is undefined for negative values
    inputs3 = torch.rand(512, 6144, dtype=torch.float32) + 0.1  # Adding 0.1 to ensure positive values
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    # Using positive values since log is undefined for negative values
    inputs4 = torch.rand(1024, 8192, dtype=torch.float32) + 0.1  # Adding 0.1 to ensure positive values
    
    return [
        [inputs1],  # Case 1 inputs
        [inputs2],  # Case 2 inputs
        [inputs3],  # Case 3 inputs
        [inputs4]   # Case 4 inputs
    ]


def get_init_inputs():
    # No parameters needed for log
    # Extract params
    return []