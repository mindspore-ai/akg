import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Split operation that splits a tensor into chunks along a specified dimension.
    This operation is commonly used in neural networks for:
    - Dividing feature maps for parallel processing
    - Implementing multi-head attention mechanisms
    - Used in architectures that require tensor partitioning
    """
    def __init__(self, split_dim=0, split_num=2):
        super(Model, self).__init__()
        self.split_dim = split_dim
        self.split_num = split_num

    def forward(self, input_tensor):
        # Split input_tensor into split_num chunks along split_dim
        result = torch.chunk(input_tensor, chunks=self.split_num, dim=self.split_dim)
        return result


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(128, 128, 1024, dtype=torch.float32)

    # Middle shape case
    input2 = torch.randn(256, 1024, 4096, dtype=torch.float32)

    # Large shape case
    input3 = torch.randn(1024, 4096, 8192, dtype=torch.float32)

    # Noaligned shape case
    input4 = torch.randn(512, 3000, 6144, dtype=torch.float32)

    return [[input1], [input2], [input3], [input4]]


def get_init_inputs():
    # Parameters for Split operation
    split_dim = 1   # Dimension along which to split
    split_num = 2    # Number of chunks to create
    return [split_dim, split_num]