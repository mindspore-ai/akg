import torch
import triton
import triton.language as tl

@triton.jit
def custom_op_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flattened 1D element-wise addition kernel.
    Works for any tensor shape (1D, 2D, 3D, ...).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b

    tl.store(c_ptr + offsets, c, mask=mask)


def custom_op_triton_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Launch flattened 1D element-wise addition kernel.
    Supports any shape, as long as x.shape == y.shape.
    """
    assert x.shape == y.shape, "Input tensors must have identical shape"
    assert x.dtype == y.dtype, "Input tensors must have same dtype"

    # Ensure contiguous memory layout for optimal performance
    x = x.contiguous()
    y = y.contiguous()
    output = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 4096
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    custom_op_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output