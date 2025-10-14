import torch
import triton
import triton.language as tl


@triton.jit
def add_dyn_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    元素级加法内核
    计算 output = input + other * alpha
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input = tl.load(input_ptr + offsets, mask=mask)
    other = tl.load(other_ptr + offsets, mask=mask)
    output = input + other * alpha
    tl.store(output_ptr + offsets, output, mask=mask)


def add_dyn_triton_torch(input_tensor, other):
    """
    内核启动器
    """
    alpha = 1.0
    output = torch.empty_like(input_tensor)
    n_elements = output.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_dyn_kernel[grid](
        input_tensor, other, output, n_elements, alpha, BLOCK_SIZE
    )

    return output
