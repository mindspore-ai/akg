import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 103}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32}),
    ],
    key=['M', 'N']
)
@triton.jit
def mean_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    out0_stride0, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    n_offsets = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    nmask = n_offsets < N

    col_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for m_start in range(0, M, BLOCK_SIZE_M):
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        mmask = m_offsets < M
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        block_vals = tl.load(block_ptrs, mask=mask, other=0.0)

        col_sum += block_vals
        
    col_sum = tl.sum(col_sum, axis=0)
    col_mean = col_sum / M

    output0_ptrs = out_ptr0 + n_offsets * out0_stride0
    tl.store(output0_ptrs, col_mean, mask=nmask)

def mean_triton_torch(input0):
    """
    2D, reduce_axis = 0
    """
    M, N = input0.shape
    output0 = torch.empty(N, device=input0.device, dtype=input0.dtype)
    
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    
    mean_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0