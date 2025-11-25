from unittest import result
import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE": 8192, "TILE_SIZE": 8192}), # 2048corenum，最优配置
    triton.Config({"BLOCK_SIZE": 4096, "TILE_SIZE": 4096}), 
    triton.Config({"BLOCK_SIZE": 2048, "TILE_SIZE": 2048}), # 
]
@triton.autotune(
    configs=configs,
    key=['n_elements'],
)
@triton.jit
def where_kernel(
    x_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    threshold_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    for i in range(0, BLOCK_SIZE, TILE_SIZE):
        offsets = block_start + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        input_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        condition = input_data > threshold_val
        result = tl.where(condition, input_data, 0.0)
        tl.store(output_ptr + offsets, result, mask=mask)

def custom_op_triton_torch(input_tensor):
    n_elements = input_tensor.numel()
    threshold_val = 0.5

    output_tensor = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    grid = lambda meta: [triton.cdiv(n_elements, meta['BLOCK_SIZE'])]
    zeros_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        threshold_val,
    )
    return output_tensor
