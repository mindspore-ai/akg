import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE": 4194304, "TILE_SIZE": 32768}),# 32corenum 性能最优！UB用满
    triton.Config({"BLOCK_SIZE": 2097152, "TILE_SIZE": 16384}),# 64corenum
    triton.Config({"BLOCK_SIZE": 1048576, "TILE_SIZE": 32768}),# 128corenum 性能较优！

]
@triton.autotune(
    configs=configs,
    key=['n_elements'],
)
@triton.jit
def cast_kernel(
    input_ptr,
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
        input_data = tl.load(input_ptr + offsets, mask=mask)
        output_data = tl.cast(input_data, tl.float32)
        tl.store(output_ptr + offsets, output_data, mask=mask)

def custom_op_triton_torch(input_tensor):
    n_elements = input_tensor.numel()
    output_tensor = torch.empty_like(input_tensor, dtype=input_tensor.dtype)
    grid = lambda meta: [triton.cdiv(n_elements, meta['BLOCK_SIZE'])]
    cast_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
    )
    return output_tensor
