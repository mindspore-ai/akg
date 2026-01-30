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


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn((128, 1024), dtype=torch.float32)
    index1 = torch.randint(0, 128, [128 // 10], dtype=torch.int64)

    # Middle shape case
    input2 = torch.randn((512, 2048), dtype=torch.float32)
    index2 = torch.randint(0, 512, [512 // 10], dtype=torch.int64)

    # Large shape case
    input3 = torch.randn((1024, 4096), dtype=torch.float32)
    index3 = torch.randint(0, 1024, [1024 // 10], dtype=torch.int64)

    # Noaligned shape case
    input4 = torch.randn((513, 3000), dtype=torch.float32)
    index4 = torch.randint(0, 513, [513 // 10], dtype=torch.int64)

    return [
        [input1, index1],
        [input2, index2],
        [input3, index3],
        [input4, index4]
    ]


def get_init_inputs():
    # Dimension along which to index
    dim = 0  # Index along the first dimension
    return [dim]