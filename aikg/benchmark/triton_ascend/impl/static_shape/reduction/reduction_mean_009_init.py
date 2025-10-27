import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(
    input_ptr, output_ptr,
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_SIZE: tl.constexpr,
):
    m_idx = tl.program_id(0)
    
    if m_idx >= M:
        return
    
    row_sum = 0.0
    
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N
        
        ptr = input_ptr + m_idx * stride_m + n_offsets * stride_n
        data = tl.load(ptr, mask=mask, other=0.0)
        
        row_sum += tl.sum(data, axis=0)

    tl.store(output_ptr + m_idx, sum_value)

def sum_triton_torch(x: torch.Tensor):
    assert x.dim() == 2
    M, N = x.shape
    output = torch.empty(M, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024

    grid = (M,)
    
    sum_kernel[grid](
        x, output,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output