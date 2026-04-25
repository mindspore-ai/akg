import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, input_tensor, index, src):
        # torch.scatter(input, dim, index, src, reduce='multiply')
        # Writes all values from the tensor src into input at the indices specified in the index tensor.
        # If reduce is 'multiply', elements in src are multiplied to the original elements in input.
        # Scatter-multiply operations are commonly used in neural networks for:
        # - Implementing specialized attention mechanisms
        # - Applying multiplicative updates
        # - Scaling specific elements in tensors
        return torch.scatter(input_tensor, self.dim, index, src, reduce='multiply')


def get_inputs_dyn_list():
    # Small shape case
    input1 = torch.randn((128, 1024), dtype=torch.float32)
    index1 = torch.randint(0, 128, (12, 1024), dtype=torch.int64)  # 128//10≈12
    src1 = torch.randn((12, 1024), dtype=torch.float32)

    # Middle shape case
    input2 = torch.randn((512, 2048), dtype=torch.float32)
    index2 = torch.randint(0, 512, (51, 2048), dtype=torch.int64)  # 512//10≈51
    src2 = torch.randn((51, 2048), dtype=torch.float32)

    # Large shape case
    input3 = torch.randn((1024, 4096), dtype=torch.float32)
    index3 = torch.randint(0, 1024, (102, 4096), dtype=torch.int64)  # 1024//10=102
    src3 = torch.randn((102, 4096), dtype=torch.float32)

    # Noaligned shape case
    input4 = torch.randn((513, 3000), dtype=torch.float32)
    index4 = torch.randint(0, 513, (51, 3000), dtype=torch.int64)  # 513//10=51
    src4 = torch.randn((51, 3000), dtype=torch.float32)

    return [
        [input1, index1, src1],
        [input2, index2, src2],
        [input3, index3, src3],
        [input4, index4, src4]
    ]


def get_init_inputs():
    # Dimension along which to scatter
    dim = 0  # Scatter along the first dimension
    return [dim]