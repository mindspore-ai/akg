import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 52}),
        triton.Config({'BLOCK_SIZE': 16}),
    ],
    key=['n_elements']
)
@triton.jit
def amin_kernel(
    in_ptr0, 
    out_ptr0, 
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    block_data = tl.load(in_ptr0 + offsets, mask=mask, other=float('inf'))

    min_val = tl.min(block_data)
    tl.atomic_min(out_ptr0, min_val)

def amin_triton_torch(input0):
    """
    1D, reduce_axis = 0
    """
    n_elements = input0.numel()
    output0 = torch.tensor([float('inf')], dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    amin_kernel[grid](
        input0, 
        output0, 
        n_elements
    )
    return output0