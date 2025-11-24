import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256}),   # 核数未用满（当前AI core为40）
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048}),   # grid等于核数，SUB切分含尾块
        triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256}),   # grid等于核数，SUB切分不含尾块
        triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 256}),   # grid超核数，且非核数整数倍
    ],
    key=['M', 'N']
)
@triton.jit
def sum_fused_elemwise_kernel(
    x_ptr, bias_ptr, output_ptr,
    M, N,
    stride_x_m, stride_x_n,
    stride_out_m,
    BLOCK_SIZE_M: tl.constexpr, 
    SUB_BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr 
):
    pid = tl.program_id(0)

    for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
        m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
        mmask = m_offsets < M

        acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
        for n_start in range(0, N, BLOCK_SIZE_N):
            n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
            nmask = n_offsets < N

            mask = (mmask[:, None]) & (nmask[None, :])
            offsets = m_offsets[:,None] * stride_x_m + n_offsets[None,:] * stride_x_n

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            bias = tl.load(bias_ptr + n_offsets, mask=nmask, other=0.0)
            
            t1 = x * 2.0
            t2 = t1 + bias
            t3 = tl.sigmoid(t2)
            
            acc += tl.where(mask, t3, 0.0)

        total_sum = tl.sum(acc, axis=1)

        output_ptrs = output_ptr + m_offsets * stride_out_m
        tl.store(output_ptrs, total_sum, mask=mmask)


def sum_fused_elemwise_triton_torch(x, bias):
    assert x.dim() == 2
    assert bias.dim() == 1
    assert x.shape[1] == bias.shape[0]
    
    M, N = x.shape
    
    output = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )
    
    sum_fused_elemwise_kernel[grid](
        x, bias, output,
        M, N,
        x.stride(0), x.stride(1),
        output.stride(0)
    )
    
    return output