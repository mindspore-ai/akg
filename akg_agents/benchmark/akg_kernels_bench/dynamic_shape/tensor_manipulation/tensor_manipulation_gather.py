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


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn(128, 1024, dtype=torch.float32)
    index1 = torch.randint(0, 1024, (128, 50), dtype=torch.int64)  # 选择50个元素

    # Middle shape case
    input2 = torch.randn(512, 2048, dtype=torch.float32)
    index2 = torch.randint(0, 2048, (512, 100), dtype=torch.int64)  # 选择100个元素

    # Large shape case
    input3 = torch.randn(1024, 4096, dtype=torch.float32)
    index3 = torch.randint(0, 4096, (1024, 100), dtype=torch.int64)  # 选择100个元素

    # Noaligned shape case
    input4 = torch.randn(513, 3000, dtype=torch.float32)
    index4 = torch.randint(0, 3000, (513, 75), dtype=torch.int64)  # 选择75个元素

    return [
        [input1, index1],
        [input2, index2],
        [input3, index3],
        [input4, index4]
    ]


def get_init_inputs():
    # Specific dim value for gathering
    # Gather along second dimension (features dimension)
    dim = 1
    return [dim]