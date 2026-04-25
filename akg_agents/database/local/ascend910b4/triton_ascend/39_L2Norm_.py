import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 16384}, num_warps=1),
    ],
    key=['B', 'D'],
)
@triton.jit
def aikg_39_L2Norm__kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    L2归一化内核
    每个程序处理一个batch
    """
    # 获取当前batch索引
    b = tl.program_id(0)
    
    # 边界检查
    if b >= B:
        return
    
    # 计算偏移量
    x_offset = b * D
    y_offset = b * D
    
    # 第一阶段：计算平方和
    sum_sq = tl.zeros([1], dtype=tl.float32)
    
    # 分块处理数据
    for n_start in range(0, D, BLOCK_SIZE_N):
        n_end = min(n_start + BLOCK_SIZE_N, D)
        actual_size = n_end - n_start
        
        # 创建偏移量
        offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < D
        
        # 加载数据
        x_tile = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        
        # 计算平方
        sq_tile = x_tile * x_tile
        
        # 累加平方和
        tile_sum = tl.sum(sq_tile, axis=0)
        sum_sq += tile_sum
    
    # 第二阶段：计算范数值
    sum_sq += eps
    norm_val = tl.math.sqrt(sum_sq)
    
    # 第三阶段：应用归一化
    for n_start in range(0, D, BLOCK_SIZE_N):
        n_end = min(n_start + BLOCK_SIZE_N, D)
        actual_size = n_end - n_start
        
        # 创建偏移量
        offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < D
        
        # 加载数据
        x_tile = tl.load(x_ptr + x_offset + offsets, mask=mask, other=0.0)
        
        # 归一化
        y_tile = x_tile / norm_val
        
        # 存储结果
        tl.store(y_ptr + y_offset + offsets, y_tile, mask=mask)


def aikg_39_L2Norm__triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    L2归一化Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，形状为 [B, D]
        
    Returns:
        torch.Tensor: 输出张量，形状为 [B, D]
    """
    # 获取输入形状
    B, D = x.shape  # B=16, D=16384
    
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 设置epsilon值
    eps = 1e-12
    
    # 启动内核
    grid = (B,)
    
    aikg_39_L2Norm__kernel[grid](
        x, y, B, D, eps
    )
    
    return y