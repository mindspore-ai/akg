import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, other):
        # torch.logical_and(input, other, *, out=None)
        # Computes the element-wise logical AND of the given input tensors.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Combining multiple boolean masks
        # - Implementing complex conditional logic
        # - Creating intersection of conditions
        return torch.logical_and(input_tensor, other)


def get_inputs_dyn_list():
    # Element-wise logical AND variation cases with both aligned and non-aligned shapes
    
    # Case 1: Small/Medium shapes
    # Shape (256, 1024) represents a batch of 256 samples with 1024 features each
    inp1_1 = torch.randint(0, 2, (256, 1024), dtype=torch.bool)
    inp2_1 = torch.randint(0, 2, (256, 1024), dtype=torch.bool)
    
    # Case 2: Standard large model shapes
    # Shape (1024, 4096) represents a batch of 1024 samples with 4096 features each
    inp1_2 = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)
    inp2_2 = torch.randint(0, 2, (1024, 4096), dtype=torch.bool)
    
    # Case 3: Large shapes
    # Shape (2048, 8192) represents a batch of 2048 samples with 8192 features each
    inp1_3 = torch.randint(0, 2, (2048, 8192), dtype=torch.bool)
    inp2_3 = torch.randint(0, 2, (2048, 8192), dtype=torch.bool)
    
    # Case 4: Non-16-aligned shapes
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inp1_4 = torch.randint(0, 2, (125, 5120), dtype=torch.bool)
    inp2_4 = torch.randint(0, 2, (125, 5120), dtype=torch.bool)
    
    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4]
    ]


def get_init_inputs():
    # No parameters needed for logical_and
    return []