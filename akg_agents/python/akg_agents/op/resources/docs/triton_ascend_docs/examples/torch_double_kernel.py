import torch
import triton
import triton.language as tl


@triton.jit
def first_kernel(input2, input3, output0):
    # 主要实现
    pass

@triton.jit
def second_kernel(input4, input5, output1):
    # 辅助实现
    pass

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input0, input1):
        """
        Triton 双kernel示例
        """
    # host侧实现
    first_kernel[grid1](input2, input3, output0)
    second_kernel[grid2](input4, input5, output1)
    return output
