import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor, other):
        # torch.logical_or(input, other, *, out=None)
        # Computes the element-wise logical OR of the given input tensors.
        # Zeros are treated as False and nonzeros are treated as True.
        # This operation is commonly used in neural networks for:
        # - Combining multiple boolean masks
        # - Implementing complex conditional logic
        # - Creating union of conditions
        return torch.logical_or(input_tensor, other)


def get_inputs_dyn_list():
    # Element-wise logical OR variation cases with both aligned and non-aligned shapes
    
    # Case 1: 16-aligned batch, 16-aligned hidden
    # Shape (256, 4096) represents a batch of 256 samples with 4096 features each
    inp1_1 = torch.randint(0, 2, (256, 4096), dtype=torch.bool)
    inp2_1 = torch.randint(0, 2, (256, 4096), dtype=torch.bool)
    
    # Case 2: Non-16-aligned batch, 16-aligned hidden
    # Shape (125, 5120) represents a batch of 125 samples with 5120 features each
    inp1_2 = torch.randint(0, 2, (125, 5120), dtype=torch.bool)
    inp2_2 = torch.randint(0, 2, (125, 5120), dtype=torch.bool)
    
    # Case 3: 16-aligned batch, non-16-aligned hidden
    # Shape (512, 6144) represents a batch of 512 samples with 6144 features each
    inp1_3 = torch.randint(0, 2, (512, 6144), dtype=torch.bool)
    inp2_3 = torch.randint(0, 2, (512, 6144), dtype=torch.bool)
    
    # Case 4: Large batch size
    # Shape (1024, 8192) represents a batch of 1024 samples with 8192 features each
    inp1_4 = torch.randint(0, 2, (1024, 8192), dtype=torch.bool)
    inp2_4 = torch.randint(0, 2, (1024, 8192), dtype=torch.bool)
    
    return [
        [inp1_1, inp2_1],
        [inp1_2, inp2_2],
        [inp1_3, inp2_3],
        [inp1_4, inp2_4]
    ]


def get_init_inputs():
    # No parameters needed for logical_or
    return []