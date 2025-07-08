import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16}, num_stages=5, num_warps=2),
    ],
    key=['N'],
)


@triton.jit
def matmul_uppertri_kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Only compute blocks in or above the diagonal
    if pid_m > pid_n:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    upper_bound = tl.minimum((pid_n + 1) * BLOCK_SIZE_N, N)
    
    for k in range(0, upper_bound, BLOCK_SIZE_K):
        k_offs = k + offs_k
        
        # Load A with proper upper triangular masking
        a_mask = (offs_m[:, None] < N) & (k_offs[None, :] < N) & (offs_m[:, None] <= k_offs[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B with proper upper triangular masking
        b_mask = (k_offs[:, None] < N) & (offs_n[None, :] < N) & (k_offs[:, None] <= offs_n[None, :])
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate with proper masking
        accumulator += tl.where(
            (offs_m[:, None] <= offs_n[None, :]),
            tl.dot(a, b),
            0.0
        )
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store only upper triangular part
    c_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N) & (offs_m[:, None] <= offs_n[None, :])
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_trimatmul(A, B):
    N = A.shape[0]
    C = torch.zeros_like(A)
    
    # Launch kernel with optimal configuration
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    matmul_uppertri_kernel[grid](
        A, B, C,
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C