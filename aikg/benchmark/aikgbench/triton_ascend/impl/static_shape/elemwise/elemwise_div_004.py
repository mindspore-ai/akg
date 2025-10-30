import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 核心用满策略 (maximize core utilization)
        # 小 BLOCK_SIZE，启动更多 blocks 接近 40 核心
        triton.Config({'BLOCK_SIZE': 32}),
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        
        # 平衡策略 (balanced)
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        
        # UB内存用满策略 (maximize memory utilization)
        # 大 BLOCK_SIZE，充分利用 192KB UB 内存
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 32768}),
    ],
    key=['n_elements'],
)
@triton.jit
def div_kernel(
    dividend_ptr,
    divisor_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the program ID (which block this is)
    pid = tl.program_id(0)
    
    # Calculate the offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data
    dividend = tl.load(dividend_ptr + offsets, mask=mask, other=0.0)
    divisor = tl.load(divisor_ptr + offsets, mask=mask, other=1.0)
    
    # Compute element-wise division
    output = dividend / divisor
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


def custom_op_triton_torch(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    # Ensure inputs are contiguous and have the same shape
    assert dividend.shape == divisor.shape, "Input tensors must have the same shape"
    assert dividend.dtype == divisor.dtype, "Input tensors must have the same dtype"
    
    dividend = dividend.contiguous()
    divisor = divisor.contiguous()
    
    # Create output tensor
    output = torch.empty_like(dividend)
    
    # Calculate total number of elements
    n_elements = dividend.numel()
    
    # Calculate grid size
    # Triton autotune 会自动选择最优的 BLOCK_SIZE
    # 这里使用一个默认值来计算grid，实际BLOCK_SIZE会被autotune优化
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel with autotune
    # Triton 会自动测试所有配置并选择最优的
    div_kernel[grid](
        dividend,
        divisor,
        output,
        n_elements,
    )
    
    return output