import torch
import triton
import triton.language as tl


@triton.jit
def clamp_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise clamp operation.
    Computes: output = clamp(input, min_val, max_val)
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements to process
        min_val: Minimum value for clamping
        max_val: Maximum value for clamping
        BLOCK_SIZE: Number of elements to process per block
    """
    # Get the program ID (which block this is)
    pid = tl.program_id(0)
    
    # Calculate the offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Compute clamp: max(min_val, min(input, max_val))
    # Equivalent to: clamp input between min_val and max_val
    output_data = tl.minimum(tl.maximum(input_data, min_val), max_val)
    
    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def custom_op_triton_torch(input_tensor: torch.Tensor, min_val: float = -1.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Triton implementation of element-wise clamp operation.
    Computes: output = clamp(input, min_val, max_val)
    
    Args:
        input_tensor: Input tensor (FP16)
        min_val: Minimum value for clamping (default: -1.0)
        max_val: Maximum value for clamping (default: 1.0)
    
    Returns:
        Output tensor with same shape as input, values clamped to [min_val, max_val]
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate total number of elements
    BLOCK_SIZE = 8
    n_elements = input_tensor.numel()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    clamp_kernel[grid](
        input_tensor,
        output,
        n_elements,
        min_val,
        max_val,
        BLOCK_SIZE
    )
    
    return output