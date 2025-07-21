import torch
import triton
import triton.language as tl

@triton.jit
def matrix_vector_multiplication__kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = offs_m < M
    
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offset = k_block * BLOCK_SIZE_K
        mask_k = (k_offset + offs_k) < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        b = tl.load(
            b_ptr + (k_offset + offs_k) * stride_bk,
            mask=mask_k,
            other=0.0
        )
        partial = tl.sum(a * b, axis=1)
        accumulator += partial
    
    tl.store(
        c_ptr + offs_m * stride_cm,
        accumulator.to(c_ptr.dtype.element_ty),  
        mask=mask_m  
    )


def matrix_vector_multiplication__triton_torch(A: torch.Tensor, B: torch.Tensor):
    """
    Launch function for standard matrix multiplication.
    Args:
        A: shape (M, K)
        B: shape (K, 1)
    Returns:
        C: Output matrix of shape (M, 1)
    """
    assert A.shape[1] == B.shape[0], "Incompatible dimensions"
    assert A.is_contiguous(), "Matrix A must be contiguous"
    
    M, K = A.shape
    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    
    BLOCK_SIZE_M = 128    
    BLOCK_SIZE_K = 64   

    # Compute grid dimensions
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), )

    # Launch kernel
    matrix_vector_multiplication__kernel[grid](
        A, B, C, 
        M, K,
        A.stride(0), A.stride(1),
        B.stride(0),
        C.stride(0),
        BLOCK_SIZE_M, BLOCK_SIZE_K
    )
    
    return C