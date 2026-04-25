import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}), 
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32}),
        triton.Config({'BLOCK_SIZE_M': 1024, 'BLOCK_SIZE_N': 16}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 112}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 96}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 80}),
    ],
    key=['M', 'N']
)
@triton.jit
def amax_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    out_stride0, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    n_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    nmask = n_offsets < N

    curr_max = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_SIZE_M):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        mmask = m_offsets < M
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=-float('inf'))
        curr_max = tl.maximum(data_block, curr_max)
    col_max = tl.max(curr_max, 0)

    output_ptrs = out_ptr0 + n_offsets * out_stride0
    tl.store(output_ptrs, col_max, mask=nmask)

def amax_triton_torch(input0):
    """
    2D, reduce_axis = 0
    """
    M, N = input0.shape
    output0 = torch.empty((N,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    amax_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0