import torch
import triton
import triton.language as tl

configs = [
    triton.Config({"BLOCK_SIZE": 65536, "TILE_SIZE": 8192}), # 2048corenum最优
]
@triton.autotune(
    configs=configs,
    key=['n_elements'],
)
@triton.jit
def full_kernel(
    x0_ptr,
    x1_ptr,
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
        x0 = tl.load(x0_ptr + offsets, mask=mask)
        x1 = tl.load(x1_ptr + offsets, mask=mask)
        ret = x0 + x1
        # 循环两次
        for j in range(2):
            ret = ret + x1
        # 循环3次
        for j in range(3):
            ret = ret + x0
        tl.store(output_ptr + offsets, ret, mask=mask)

def custom_op_triton_torch(x0, x1):
    L, M, N = x0.shape
    dtype = x0.dtype
    n_elements = L * M * N

    output_tensor = torch.empty_like(x0)
    x0_flat = x0.reshape(-1)
    x1_flat = x1.reshape(-1)
    output_flat = output_tensor.reshape(-1)

    grid = lambda meta: [triton.cdiv(n_elements, meta['BLOCK_SIZE'])]
    full_kernel[grid](
        x0_flat,
        x1_flat,
        output_flat,
        n_elements,
    )
    return output_tensor.reshape(L, M, N)
