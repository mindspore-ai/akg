import torch
import triton
import triton.language as tl
import mindspore as ms


@triton.jit
def first_kernel(input2, input3, output0):
    # 主要实现
    pass


@triton.jit
def second_kernel(input4, input5, output1):
    # 辅助实现
    pass


def host_triton_torch(input0, input1):
    # host侧实现
    first_kernel[grid1](input2, input3, output0)

    second_kernel[grid2](input4, input5, output1)

    return output
