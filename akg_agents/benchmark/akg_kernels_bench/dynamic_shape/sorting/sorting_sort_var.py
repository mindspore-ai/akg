import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_tensor):
        # torch.sort(input, dim=-1, descending=False, stable=False, out=None)
        # Sorts the elements of the input tensor along a given dimension in ascending or descending order.
        # This operation is commonly used in neural networks for:
        # - Sorting attention weights to implement sparse attention
        # - Finding top-k elements in recommendation systems
        # - Implementing ranking algorithms
        return torch.sort(input_tensor, dim=-1, descending=False)


def get_inputs_dyn_list():
    # Case 1: Small batch, small hidden (non-aligned batch)
    input_tensor1 = torch.randn(15, 1344, dtype=torch.float32)
    
    # Case 2: Small batch, large hidden (aligned batch)
    input_tensor2 = torch.randn(16, 4096, dtype=torch.float32)
    
    # Case 3: Medium batch, medium hidden (non-aligned batch)
    input_tensor3 = torch.randn(127, 2688, dtype=torch.float32)
    
    # Case 4: Large batch, large hidden (aligned batch)
    input_tensor4 = torch.randn(512, 5120, dtype=torch.float32)
    
    # Case 5: Very large batch, very large hidden (non-aligned batch)
    input_tensor5 = torch.randn(1023, 8192, dtype=torch.float32)
    
    return [
        [input_tensor1],
        [input_tensor2],
        [input_tensor3],
        [input_tensor4],
        [input_tensor5]
    ]


def get_init_inputs():
    # No parameters needed for sort
    return []