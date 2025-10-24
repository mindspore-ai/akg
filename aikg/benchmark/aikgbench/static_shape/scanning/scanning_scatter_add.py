import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index, src):
        # torch.scatter(input, dim, index, src, reduce='add')
        # Writes all values from the tensor src into input at the indices specified in the index tensor.
        # If reduce is 'add', elements in src are added to the original elements in input.
        # Scatter-add operations are commonly used in neural networks for:
        # - Implementing attention mechanisms
        # - Accumulating gradients in sparse operations
        # - Updating embedding tables
        return torch.scatter(input_tensor, self.dim, index, src, reduce='add')


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    input_tensor = torch.randn((1024, 4096), dtype=torch.float32)
    
    # Create index tensor for scattering
    index_shape = (102, 4096)  # Smaller index tensor (1024//10, 4096)
    index = torch.randint(0, 1024, index_shape, dtype=torch.long)
    
    # Source tensor with the same shape as index
    src = torch.randn(index_shape, dtype=torch.float32)
    
    return [input_tensor, index, src]


def get_init_inputs():
    # Dimension along which to scatter
    dim = 0  # Scatter along the first dimension
    return [dim]