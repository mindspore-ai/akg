import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE': 2048, 'VEC_SIZE': 256}),
        triton.Config({'TILE_SIZE': 1024, 'VEC_SIZE': 256}),
        triton.Config({'TILE_SIZE': 512, 'VEC_SIZE': 256}),
        triton.Config({'TILE_SIZE': 256, 'VEC_SIZE': 256}),
        triton.Config({'TILE_SIZE': 2048, 'VEC_SIZE': 128}),
        triton.Config({'TILE_SIZE': 1024, 'VEC_SIZE': 128}),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_93_masked_cumsum_kernel(
    x_ptr,
    mask_ptr,
    output_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Triton masked cumulative sum kernel
    Computes cumsum(x * mask) along the last dimension
    """
    # Get batch index
    b_idx = tl.program_id(0)
    
    # Initialize accumulator
    acc = 0.0
    
    # Process in tiles
    for i_outer in range(0, N, TILE_SIZE):
        tile_end = min(i_outer + TILE_SIZE, N)
        current_tile_size = tile_end - i_outer
        
        # Process tile in vector chunks
        for i_inner in range(0, current_tile_size, VEC_SIZE):
            vec_end = min(i_inner + VEC_SIZE, current_tile_size)
            vec_size = vec_end - i_inner
            
            # Calculate offsets
            offsets = i_outer + i_inner + tl.arange(0, VEC_SIZE)
            mask = offsets < N
            
            # Load data with mask
            x_tile = tl.load(x_ptr + b_idx * N + offsets, mask=mask, other=0.0)
            mask_tile = tl.load(mask_ptr + b_idx * N + offsets, mask=mask, other=False)
            
            # Convert boolean mask to float32
            mask_float = mask_tile.to(tl.float32)
            
            # Apply mask: x_masked = x * mask
            x_masked = x_tile * mask_float
            
            # Process each element in the vector
            for j in range(vec_size):
                if j < vec_size:
                    element_val = tl.get_element(x_masked, (j,))
                    acc += element_val
                    
                    # Calculate output index
                    output_idx = b_idx * N + i_outer + i_inner + j
                    
                    # Store result
                    tl.store(output_ptr + output_idx, acc, mask=(output_idx < B * N))


def aikg_93_masked_cumsum_triton_ascend_torch(x, mask, dim=1):
    """
    Triton implementation of masked cumulative sum
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, *input_shape)
        mask (torch.Tensor): Boolean mask of the same shape as x
        dim (int): Dimension along which to perform cumulative sum
    
    Returns:
        torch.Tensor: Cumulative sum of elements where mask is True
    """
    # Ensure inputs are contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    if not mask.is_contiguous():
        mask = mask.contiguous()
    
    # Get shape parameters
    B = x.shape[0]  # batch_size = 128
    N = x.shape[1]  # input_shape[0] = 4000
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Calculate grid size
    grid = (B,)
    
    # Launch kernel
    aikg_93_masked_cumsum_kernel[grid](
        x, mask, output, B, N
        # TILE_SIZE and VEC_SIZE are automatically provided by autotune
    )
    
    return output