import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 16}),
        # triton.Config({'BLOCK_SIZE_N': 512, 'BLOCK_SIZE_M': 8}),
        # triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64}),
        # triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 128}),
    ],
    key=['M', 'N'],
)
@triton.jit
def aikg_53_Min_reduction_over_a_dimension_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    """
    Min reduction over dimension 1 (M dimension) of a 3D tensor [B, M, N]
    Output: [B, N]
    """
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Calculate batch and n block indices
    b_idx = pid // num_blocks_n
    n_block_idx = pid % num_blocks_n
    n_start = n_block_idx * BLOCK_SIZE_N
    
    # Initialize accumulator with maximum float32 value
    min_acc = tl.full([BLOCK_SIZE_N], float('inf'), dtype=tl.float32)
    
    # Define n_offsets outside the loop for later use
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # Process M dimension in blocks
    for m_start in range(0, M, BLOCK_SIZE_M):
        # Create offsets for current tile
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        
        # Create 2D mask for current tile
        m_mask = m_offsets < M
        tile_mask = m_mask[:, None] & n_mask[None, :]
        
        # Calculate pointer offsets
        x_offset = (b_idx * stride_xb + 
                   m_offsets[:, None] * stride_xm + 
                   n_offsets[None, :] * stride_xn)
        
        # Load data tile
        x_tile = tl.load(x_ptr + x_offset, mask=tile_mask, other=float('inf'))
        
        # Reduce min over M dimension for current tile
        block_min = tl.min(x_tile, axis=0)
        
        # Update global min accumulator
        min_acc = tl.minimum(min_acc, block_min)
    
    # Store result
    y_offset = b_idx * stride_yb + n_offsets * stride_yn
    tl.store(y_ptr + y_offset, min_acc, mask=n_mask)


def aikg_53_Min_reduction_over_a_dimension_triton_ascend_torch(x: torch.Tensor):
    """
    Triton implementation of min reduction over dimension 1
    
    Args:
        x (torch.Tensor): Input tensor of shape [B, M, N] (B=16, M=256, N=256)
        
    Returns:
        torch.Tensor: Output tensor of shape [B, N] containing min values
    """
    # Get input shape parameters
    B, M, N = x.shape  # B=16, M=256, N=256
    
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Create output tensor
    y = torch.empty((B, N), dtype=x.dtype, device=x.device)
    
    # Calculate number of blocks
    grid = lambda meta: (B * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    
    # Launch kernel
    aikg_53_Min_reduction_over_a_dimension_kernel[grid](
        x, y, B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1),
    )
    
    return y