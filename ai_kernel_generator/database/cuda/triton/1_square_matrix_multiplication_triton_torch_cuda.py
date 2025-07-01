import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['N'],
)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Matrix pointers: a(N, N) * b(N, N) = c(N, N)
    N,                    # Matrix dimension (square matrices)
    stride_am, stride_ak, # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    # Compute program ID
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_n - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for blocks
    offs_m = pid_m * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_N)
    
    # Pointers to blocks in A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Compute block matrix multiplication
    for k in range(0, N, BLOCK_SIZE_N):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < N - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < N - k, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_N * stride_ak
        b_ptrs += BLOCK_SIZE_N * stride_bk
    
    # Store result
    c = accumulator.to(tl.float16)
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < N) & (offs_n[None, :] < N))
    
    
def triton_matmul(a, b, activation=""):
    N = a.shape[0]
    
    c = torch.empty((N, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c