import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.randint_like(input, low, high, *, dtype=None, layout=torch.strided, device=None, requires_grad=False, memory_format=torch.preserve_format)
        # Returns a tensor with the same shape as Tensor input filled with random integers generated uniformly.
        # This operation is commonly used in neural networks for:
        # - Generating random indices
        # - Creating random integer tensors for operations
        # - Implementing random sampling operations
        return torch.randint_like(input_tensor, 0, 100)


def get_inputs_dyn_list():
    # Random integer like variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inputs1 = torch.randn(256, 4096, dtype=torch.float32)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inputs2 = torch.randn(125, 5120, dtype=torch.float32)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inputs3 = torch.randn(512, 6144, dtype=torch.float32)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inputs4 = torch.randn(1024, 8192, dtype=torch.float32)
    
    return [
        [inputs1],
        [inputs2],
        [inputs3],
        [inputs4]
    ]


def get_init_inputs():
    # No parameters needed for randint_like
    return []