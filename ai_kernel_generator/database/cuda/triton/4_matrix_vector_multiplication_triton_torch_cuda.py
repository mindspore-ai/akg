import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 4096}, num_stages=5, num_warps=2),
    ],
    key=['M', 'K'],
)


@triton.jit
def matvecmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Create ranges for blocks
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Mask for boundary checks
    mask_m = offs_m < M
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
    
    for k_block in range(0, num_k_blocks):
        k_offset = k_block * BLOCK_SIZE_K
        # Create mask for K dimension
        mask_k = (k_offset + offs_k) < K
        
        # Load block from A
        a_vals = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k_offset + offs_k[None, :]) * stride_ak,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        
        # Load block from B
        b_vals = tl.load(
            b_ptr + (k_offset + offs_k) * stride_bk,
            mask=mask_k,
            other=0.0
        )
        
        # Compute partial dot product
        partial = tl.sum(a_vals * b_vals, axis=1)
        accumulator += partial
    
    
    # Convert and store result
    tl.store(
        c_ptr + offs_m * stride_cm,
        accumulator.to(c_ptr.dtype.element_ty),  # Use the output tensor's dtype
        mask=mask_m  # Remove the extra dimension from the mask
    )
    
    
def triton_matvecmul(a, b, activation=""):
    M, K = a.shape
    
    # Make sure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous().reshape(-1)  # Flatten b to (K,) shape
    
    # Allocate output
    c = torch.empty(M, device=a.device, dtype=a.dtype)
    
    # Calculate grid size
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    
    # Launch kernel
    matvecmul_kernel[grid](
        a, b, c,
        M, K,
        a.stride(0), a.stride(1),
        b.stride(0),
        c.stride(0),
        ACTIVATION=activation,
    )
    return c.unsqueeze(-1)