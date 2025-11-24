import torch
import triton
import triton.language as tl

@triton.jit
def mul(a, b):
    return a * b

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_M': 26, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
    ],
    key=['M', 'N']
)
@triton.jit
def prod_kernel(
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

    row_prod = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), 1.0, dtype=tl.float32)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])

        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        block_vals = tl.load(block_ptrs, mask=mask, other=1.0)

        row_prod *= block_vals
    
    row_prod = tl.reduce(row_prod, axis=1, combine_fn=mul)
    
    output0_ptrs = out_ptr0 + m_offsets * out0_stride0
    tl.store(output0_ptrs, row_prod, mask=mmask)


def prod_triton_torch(input0):
    """
    2D, reduce_axis = 1
    """
    M, N = input0.shape
    output0 = torch.empty((M,), dtype=input0.dtype, device=input0.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

    prod_kernel[grid](
        input0, 
        output0, 
        input0.stride(0),
        input0.stride(1),
        output0.stride(0),
        M, 
        N
    )
    return output0