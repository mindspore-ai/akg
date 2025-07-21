import torch
import triton
import triton.language as tl

@triton.jit
def square_matrix_multiplication__kernel(
    a_ptr, b_ptr, c_ptr,
    N,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    m_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
    b_ptrs = b_ptr + (k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        a_mask = (m_offsets[:, None] < N) & (k_offsets[None, :] < N - k * BLOCK_SIZE_K)
        b_mask = (k_offsets[:, None] < N - k * BLOCK_SIZE_K) & (n_offsets[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Store result
    c_ptrs = c_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    c_mask = (m_offsets[:, None] < N) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def square_matrix_multiplication__triton_torch(A: torch.Tensor, B: torch.Tensor):
    """
    Launch function for square matrix multiplication.
    Args:
        A: square matrix of shape (N, N)
        B: square matrix of shape (N, N)
    Returns:
        C: Output matrix of shape (N, N)
    """
    assert A.shape == B.shape, "Input matrices must have the same shape"
    assert A.shape[0] == A.shape[1], "Input matrices must be square"
    
    N = A.shape[0]
    C = torch.empty_like(A)
    
    # Define block sizes (from AUL tiling parameters)
    BLOCK_SIZE_M = 256  
    BLOCK_SIZE_N = 32   
    BLOCK_SIZE_K = 32   
    GROUP_SIZE_M = 8

    # Compute grid dimensions
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )

    # Launch kernel
    square_matrix_multiplication__kernel[grid](
        A, B, C, 
        N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        GROUP_SIZE_M
    )
    
    return C