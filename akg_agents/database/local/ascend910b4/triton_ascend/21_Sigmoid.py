import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192, 'VEC_SIZE': 512}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_SIZE': 256}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_SIZE': 128}, num_warps=1),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_21_Sigmoid_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr = 20,
):
    """
    Sigmoid激活函数Triton内核实现
    
    Args:
        x_ptr: 输入张量指针
        y_ptr: 输出张量指针
        B: batch维度大小
        N: 特征维度大小
        BLOCK_SIZE: 块大小
        VEC_SIZE: 向量大小
        NUM_CORES: AI Core数量
    """
    pid = tl.program_id(0)  # 核心ID: 0~NUM_CORES-1
    
    # 计算每行需要的块数
    blocks_per_row = tl.cdiv(N, BLOCK_SIZE)
    total_blocks = B * blocks_per_row
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, NUM_CORES):
        # 计算当前块的2D索引
        b_idx = block_idx // blocks_per_row  # batch索引
        n_block = block_idx % blocks_per_row  # 列块索引
        
        # 计算当前块的起始和结束位置
        n_start = n_block * BLOCK_SIZE
        n_end = min(n_start + BLOCK_SIZE, N)
        
        # 计算偏移和掩码
        offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_end
        
        # 计算输入指针偏移
        x_row_ptr = x_ptr + b_idx * N
        y_row_ptr = y_ptr + b_idx * N
        
        # 向量化加载数据
        x_tile = tl.load(x_row_ptr + offsets, mask=mask, other=0.0)
        
        # Sigmoid计算: 1 / (1 + exp(-x))
        neg_x = -x_tile
        exp_neg_x = tl.exp(neg_x)
        one_plus_exp = 1.0 + exp_neg_x
        y_tile = 1.0 / one_plus_exp
        
        # 向量化存储结果
        tl.store(y_row_ptr + offsets, y_tile, mask=mask)


def aikg_21_Sigmoid_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoid激活函数Triton实现
    
    Args:
        x: 输入张量，形状为[B, N]
        
    Returns:
        torch.Tensor: Sigmoid激活后的输出张量，形状与输入相同
    """
    # 获取输入形状参数
    B, N = x.shape  # B=16, N=16384
    
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 定义grid函数（使用lambda表达式）
    grid = lambda meta: (meta['NUM_CORES'],)
    
    # 启动内核
    aikg_21_Sigmoid_kernel[grid](
        x, y, B, N,
        # BLOCK_SIZE和VEC_SIZE由autotune自动传入
    )
    
    return y