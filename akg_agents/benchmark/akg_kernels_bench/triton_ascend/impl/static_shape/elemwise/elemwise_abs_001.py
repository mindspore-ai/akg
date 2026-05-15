import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # 2D 切分：简单版本，无核内切分
        # 行优先切分
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 1024}),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512}),

    ],
    key=['M', 'N'],
)
@triton.jit
def abs_kernel_2d(
    input_ptr,
    output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid，每个 block 处理 (BLOCK_M, BLOCK_N) 区域
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算当前 block 的行列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 创建掩码
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # 2D 索引
    offs_m_2d = offs_m[:, None]
    offs_n_2d = offs_n[None, :]
    
    # 计算全局偏移
    offs = offs_m_2d * N + offs_n_2d
    
    # 2D 掩码
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    # 加载数据
    input_data = tl.load(input_ptr + offs, mask=mask_2d, other=0)
    
    # 计算：取绝对值
    output_data = tl.abs(input_data)
    
    # 存储
    tl.store(output_ptr + offs, output_data, mask=mask_2d)


def custom_op_triton_torch(input_tensor: torch.Tensor) -> torch.Tensor:
    input_tensor = input_tensor.contiguous()
    output = torch.empty_like(input_tensor)
    
    M, N = input_tensor.shape
    
    # 2D grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    abs_kernel_2d[grid](
        input_tensor, output,
        M, N,
    )
    
    return output