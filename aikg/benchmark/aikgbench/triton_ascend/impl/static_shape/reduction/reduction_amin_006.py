import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE': 262144, 'SUB_BLOCK_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 104858, 'SUB_BLOCK_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 65536, 'SUB_BLOCK_SIZE': 32768}),
    ],
    key=['n_elements']
)
@triton.jit
def amin_kernel(
    in_ptr0, 
    out_ptr0, 
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    curr_min = tl.full((SUB_BLOCK_SIZE,), float('inf'), dtype=tl.float32)
    for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
        offsets = pid * BLOCK_SIZE + start + tl.arange(0, SUB_BLOCK_SIZE)
        mask = offsets < n_elements

        block_data = tl.load(in_ptr0 + offsets, mask=mask, other=float('inf'))
        curr_min = tl.minimum(curr_min, block_data)

    min_val = tl.min(curr_min)
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