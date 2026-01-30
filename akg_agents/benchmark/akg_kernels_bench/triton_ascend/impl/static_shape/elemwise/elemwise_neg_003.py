import torch
import triton
import triton.language as tl


@triton.jit
def neg_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for element-wise negation.
    Computes: output = -input
    
    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Total number of elements to process
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
    
    # Compute negation
    output_data = -input_data
    
    # Store the result
    tl.store(output_ptr + offsets, output_data, mask=mask)


def custom_op_triton_torch(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation of element-wise negation.
    Computes: output = -input
    
    Args:
        input_tensor: Input tensor (FP32)
    
    Returns:
        Output tensor with same shape as input
    """
    # Ensure input is contiguous
    input_tensor = input_tensor.contiguous()
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate total number of elements
    n_elements = input_tensor.numel()
    
    # Choose block size
    BLOCK_SIZE = 4
    
    # Calculate grid size (number of blocks needed)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    neg_kernel[grid](
        input_tensor,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output