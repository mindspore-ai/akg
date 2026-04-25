import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192}),
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096}),
    ],
    key=['M', 'N']
)
@triton.jit
def sum_kernel(
    in_ptr0, out_ptr0, 
    in_stride0, in_stride1, 
    out0_stride0, 
    M, 
    N,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M

    row_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        block_vals = tl.load(block_ptrs, mask=mask, other=0.0)

        row_sum += block_vals
        
    row_sum = tl.sum(row_sum, axis=1)
    output0_ptrs = out_ptr0 + m_offsets * out0_stride0
    tl.store(output0_ptrs, row_sum, mask=mmask)


def sum_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.empty((M,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

    sum_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0