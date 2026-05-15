import torch
import triton
import triton.language as tl

@triton.jit
def custom_op_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)
    b_val = tl.load(b_ptr)  # load b[0] once
    
    tl.store(c_ptr + offsets, a + b_val, mask=mask)


def custom_op_triton_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == (128,) and y.shape == (1,), "Only support this broadcast pattern for now"
    
    x = x.contiguous()
    y = y.contiguous()

    output = torch.empty_like(x)
    
    N = x.numel()
    BLOCK_SIZE = 32  # small scale
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    custom_op_kernel[grid](
        x, y, output,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output