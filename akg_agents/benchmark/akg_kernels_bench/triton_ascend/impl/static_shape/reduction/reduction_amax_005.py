import torch
import triton
import triton.language as tl

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16384}),
        triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192}), 
        triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096}),
        triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048}),
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
    row_max = tl.max(curr_max, 1)

    output_ptrs = out_ptr0 + m_offsets * out_stride0
    tl.store(output_ptrs, row_max, mask=mmask)

def amax_triton_torch(input0):
    """
    3D, reduce_axis = (1, 2)
    """
    M, K, N = input0.shape
    input0_flat = input0.reshape(M, -1)
    output0 = torch.empty(M, device=input0.device, dtype=input0.dtype)
    
    # 计算网格大小
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    
    amax_kernel[grid](
        input0_flat, output0,
        M, K * N,
        input0_flat.stride(0),
        input0_flat.stride(1),
        output0.stride(0)
    )
    return output0