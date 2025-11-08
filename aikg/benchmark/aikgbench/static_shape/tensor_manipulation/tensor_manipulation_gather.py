import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim=None):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index):
        # torch.gather(input, dim, index, *, sparse_grad=False, out=None)
        # Gathers values along an axis specified by dim.
        # For each element in the output tensor, it takes the value from the input tensor at the position
        # specified by the corresponding element in the index tensor.
        # This operation is commonly used in neural networks for:
        # - Implementing attention mechanisms
        # - Performing embedding lookups
        # - Selecting specific elements based on indices
        return torch.gather(input_tensor, dim=self.dim, index=index)


def get_inputs():
    # Batch size: 1024
    # Hidden dimension: 4096
    # Index shape (1024, 100) represents selecting 100 elements from each sample
    input_tensor = torch.randn(1024, 4096, dtype=torch.float32)
    index = torch.randint(0, 4096, (1024, 100), dtype=torch.long)
    return [input_tensor, index]


def get_init_inputs():
    # Specific dim value for gathering
    # Gather along second dimension (features dimension)
    dim = 1
    return [dim]