import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=-1):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index, src):
        # torch.scatter(input, dim, index, src, *, reduce=None, out=None)
        # Writes all values from the tensor src into input at the indices specified in the index tensor.
        # This operation is commonly used in neural networks for:
        # - Updating specific positions in tensors
        # - Implementing sparse operations
        # - Applying positional encodings
        return torch.scatter(input_tensor, self.dim, index, src)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Index shape (1024, 100) represents updating 100 elements in each sample
    # Source shape (1024, 100) represents the values to scatter
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    index = torch.randint(0, 4096, (1024, 100), dtype=torch.long)
    src = torch.randn(1024, 100, dtype=torch.float32)
    return [input_tensor, index, src]


def get_init_inputs():
    # Parameters needed for scatter
    dim = -1  # Scatter dimension
    return [dim]