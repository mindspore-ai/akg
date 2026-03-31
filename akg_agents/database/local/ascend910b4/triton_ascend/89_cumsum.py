import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE': 512, 'VEC_SIZE': 16}),
        triton.Config({'TILE_SIZE': 1024, 'VEC_SIZE': 32}),
        triton.Config({'TILE_SIZE': 2048, 'VEC_SIZE': 64}),
        triton.Config({'TILE_SIZE': 256, 'VEC_SIZE': 8}),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_89_cumsum_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Triton cumulative sum kernel along dimension 1
    """
    # Get batch index
    b_idx = tl.program_id(0)
    
    # Check if within bounds
    if b_idx >= B:
        return
    
    # Initialize accumulator as scalar
    acc = 0.0
    
    # Calculate base pointer for this batch
    x_batch_ptr = x_ptr + b_idx * stride_b
    y_batch_ptr = y_ptr + b_idx * stride_b
    
    # Process in tiles
    for start_n in range(0, N, TILE_SIZE):
        end_n = min(start_n + TILE_SIZE, N)
        current_tile_size = end_n - start_n
        
        # Process vector by vector
        for i in range(0, current_tile_size, VEC_SIZE):
            vec_end = min(i + VEC_SIZE, current_tile_size)
            vec_size = vec_end - i
            
            # Create mask for vector load
            vec_mask = tl.arange(0, VEC_SIZE) < vec_size
            
            # Load vector values
            vec_offset = start_n + i + tl.arange(0, VEC_SIZE)
            vec_vals = tl.load(x_batch_ptr + vec_offset, mask=vec_mask, other=0.0)
            
            # Process each element in the vector
            for j in range(VEC_SIZE):
                # Check if element is valid
                if j < vec_size:
                    # Get element value using tl.get_element
                    val = tl.get_element(vec_vals, (j,))
                    # Accumulate
                    acc += val
                    # Store result
                    store_offset = start_n + i + j
                    tl.store(y_batch_ptr + store_offset, acc, mask=(store_offset < N))


def aikg_89_cumsum_triton_ascend_torch(x, dim=1):
    """
    Triton cumulative sum implementation for torch
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, N)  # B=128, N=4000
        dim (int): Dimension along which to compute cumulative sum
        
    Returns:
        torch.Tensor: Cumulative sum result
    """
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Get shape parameters
    B, N = x.shape  # B=128, N=4000
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Calculate strides
    stride_b = x.stride(0)
    stride_n = x.stride(1)
    
    # Define grid - ensure grid size <= 65535
    grid = (min(B, 65535),)
    
    # Launch kernel
    aikg_89_cumsum_kernel[grid](
        x, y, B, N, stride_b, stride_n
    )
    
    return y