import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index):
        # torch.index_select(input, dim, index, *, out=None)
        # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index.
        # Index selection is commonly used in neural networks for:
        # - Gathering specific elements from tensors
        # - Implementing embedding lookups
        # - Selecting specific features or samples
        return torch.index_select(input_tensor, self.dim, index)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn((1024, 4096), dtype=torch.float32)
    
    # Create index tensor selecting about 10% of the elements along dimension 0
    index_size = input_tensor.size(0)
    index = torch.randint(0, index_size, [index_size // 10], dtype=torch.long)
    
    return [input_tensor, index]


def get_init_inputs():
    # Dimension along which to index
    dim = 0  # Index along the first dimension
    return [dim]