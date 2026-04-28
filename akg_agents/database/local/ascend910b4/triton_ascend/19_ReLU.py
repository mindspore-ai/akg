import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CORES': 16}),
        triton.Config({'BLOCK_SIZE': 16384, 'NUM_CORES': 8}),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_19_ReLU_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    stride_b: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    Triton ReLU 内核
    每个AI Core处理多行数据，每行分块处理
    """
    # 获取当前核心ID
    core_id = tl.program_id(0)
    
    # 每个核心循环处理分配的行
    for b in range(core_id, B, NUM_CORES):
        # 计算当前行的起始指针
        row_start = b * stride_b
        
        # 分块处理当前行的数据
        for block_start in range(0, N, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, N)
            block_len = block_end - block_start
            
            # 计算当前块的偏移
            offsets = row_start + block_start + tl.arange(0, BLOCK_SIZE)
            mask = tl.arange(0, BLOCK_SIZE) < block_len
            
            # 加载数据到UB
            x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
            
            # 向量化ReLU计算: max(0, x)
            y_data = tl.maximum(x_data, 0.0)
            
            # 存储结果到GM
            tl.store(y_ptr + offsets, y_data, mask=mask)


def aikg_19_ReLU_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Triton ReLU 启动函数
    
    Args:
        x (torch.Tensor): 输入张量，形状为 [B, N]
        
    Returns:
        torch.Tensor: 输出张量，形状与输入相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状参数
    B, N = x.shape  # B=16, N=16384
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    grid = lambda meta: (meta['NUM_CORES'],)
    
    # 启动内核
    aikg_19_ReLU_kernel[grid](
        x, y, B, N, 
        x.stride(0), x.stride(1),
        # BLOCK_SIZE和NUM_CORES由autotune自动传入
    )
    
    return y