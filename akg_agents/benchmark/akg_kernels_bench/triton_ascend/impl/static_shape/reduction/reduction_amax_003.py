import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 1024}),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 26, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}),
    ],
    key=['M', 'N']
)
@triton.jit
def amax_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    curr_max = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=-float('inf'))
        curr_max = tl.maximum(data_block, curr_max)
    max_val = tl.max(curr_max)
    tl.atomic_max(out_ptr0, max_val)

def amax_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.tensor([-float('inf')], dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

    amax_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        M, 
        N
    )
    return output0