import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512}),   # 核数未用满（当前AI core为40）
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096}),   # grid等于核数，SUB切分含尾块
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512}),   # grid等于核数，SUB切分不含尾块
        triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 512}),   # grid超核数，且非核数整数倍
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
    SUB_BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
        m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
        mmask = m_offsets < M

        row_sum = tl.zeros((SUB_BLOCK_SIZE_M, ), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_SIZE_N):
            n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
            nmask = n_offsets < N
            mask = (mmask[:, None]) & (nmask[None, :])

            block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
            block_vals = tl.load(block_ptrs, mask=mask, other=0.0)

            row_sum += tl.sum(block_vals, axis=1)

        row_mean = row_sum / N

        output0_ptrs = out_ptr0 + m_offsets * out0_stride0
        tl.store(output0_ptrs, row_mean, mask=mmask)


def mean_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.empty((M,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

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