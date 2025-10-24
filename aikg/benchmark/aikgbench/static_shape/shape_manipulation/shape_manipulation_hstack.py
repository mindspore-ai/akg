import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var, var_scale, indices):
        # torch.hstack(tensors, *, out=None)
        # Stack tensors in sequence horizontally (column wise).
        # This is equivalent to concatenation along the first axis for 1-D tensors,
        # and along the second axis for 2-D tensors.
        # Horizontal stacking is commonly used in neural networks for:
        # - Combining feature vectors
        # - Creating wider layers
        tensors = [var, var_scale, indices]
        return torch.hstack(tensors)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Creating 3 tensors to hstack
    tensor1 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor2 = torch.randn(1024, 4096, dtype=torch.float32)
    tensor3 = torch.randn(1024, 4096, dtype=torch.float32)
    
    # Return 3 tensors to match forward method parameters
    return [tensor1, tensor2, tensor3]


def get_init_inputs():
    # No parameters needed for hstack
    # Extract params
    return []
