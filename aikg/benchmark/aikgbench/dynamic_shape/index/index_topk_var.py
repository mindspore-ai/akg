import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, k):
        super(Model, self).__init__()
        self.k = k

    def forward(self, input_tensor):
        # torch.topk(input, k, dim=-1, largest=True, sorted=True, *, out=None)
        # Returns the k largest elements of the given input tensor along a given dimension.
        # Returns a namedtuple (values, indices) where values is the k largest elements
        # and indices is the indices of the k largest elements in the original tensor.
        # Top-k operations are commonly used in neural networks for:
        # - Implementing attention mechanisms
        # - Selecting the most probable tokens in language models
        # - Non-maximum suppression in object detection
        return torch.topk(input_tensor, self.k)


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
    # Parameters for topk
    k = 5  # Number of top elements to return
    return [k]