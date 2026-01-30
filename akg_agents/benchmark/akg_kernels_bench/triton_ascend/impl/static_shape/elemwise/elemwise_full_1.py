import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE": 8192, "TILE_SIZE": 8192}),
    triton.Config({"BLOCK_SIZE": 4096, "TILE_SIZE": 4096}), # 最优，这里使用了切分，尽量利用UB
    triton.Config({"BLOCK_SIZE": 2048, "TILE_SIZE": 2048}),
    triton.Config({"BLOCK_SIZE": 1024, "TILE_SIZE": 1024}),
]
@triton.autotune(
    configs=configs,
    key=['n_elements'],
)
@triton.jit
def full_kernel(
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    for i in range(0, BLOCK_SIZE, TILE_SIZE):
        offsets = block_start + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        fill_value = tl.full((TILE_SIZE,), 100, dtype=tl.float32)
        tl.store(output_ptr + offsets, fill_value, mask=mask)

def custom_op_triton_torch(input_tensor):
    n_elements = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor)
    grid = lambda meta: [triton.cdiv(n_elements, meta['BLOCK_SIZE'])]
    full_kernel[grid](
        output_tensor,
        n_elements,
    )
    return output_tensor
