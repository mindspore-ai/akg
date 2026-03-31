import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_CORES': 40}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_CORES': 40}),
    ],
    key=['feature_size'],
)
@triton.jit
def aikg_40_LayerNorm_kernel(
    X_ptr,  # 输入张量指针
    Y_ptr,  # 输出张量指针
    batch_size: tl.constexpr,  # 批次大小
    feature_size: tl.constexpr,  # 特征维度大小
    eps: tl.constexpr,  # epsilon值
    BLOCK_SIZE: tl.constexpr,  # 块大小
    NUM_CORES: tl.constexpr,  # 核心数
):
    """
    LayerNorm Triton 内核实现
    每个核心处理一个批次的数据
    """
    # 获取核心ID
    core_id = tl.program_id(0)
    
    # 每个核心处理多个批次
    for batch_idx in range(core_id, batch_size, NUM_CORES):
        # 计算当前批次的偏移量
        batch_offset = batch_idx * feature_size
        
        # 第一阶段：计算均值和方差
        mean_accumulator = 0.0
        var_accumulator = 0.0
        
        # 分块处理特征维度
        for i in range(0, feature_size, BLOCK_SIZE):
            # 计算当前块的偏移和掩码
            offsets = batch_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_offset + feature_size
            
            # 加载数据块
            x_tile = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            
            # 累加求和
            tile_sum = tl.sum(x_tile, axis=0)
            mean_accumulator += tile_sum
            
            # 计算平方和
            squared_tile = x_tile * x_tile
            tile_sum_squared = tl.sum(squared_tile, axis=0)
            var_accumulator += tile_sum_squared
        
        # 计算最终统计量
        mean_val = mean_accumulator / feature_size
        var_val = var_accumulator / feature_size - mean_val * mean_val
        std_val = tl.sqrt(var_val + eps)
        
        # 第二阶段：归一化计算
        for i in range(0, feature_size, BLOCK_SIZE):
            # 计算当前块的偏移和掩码
            offsets = batch_offset + i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_offset + feature_size
            
            # 加载输入数据
            x_tile = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            
            # 归一化计算
            y_tile = (x_tile - mean_val) / std_val
            
            # 存储结果
            tl.store(Y_ptr + offsets, y_tile, mask=mask)


def aikg_40_LayerNorm_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    LayerNorm Triton 启动函数
    
    Args:
        x: 输入张量，形状为 [B, F, D1, D2]
        
    Returns:
        输出张量，形状与输入相同
    """
    # 获取输入形状参数
    B, F, D1, D2 = x.shape  # B=16, F=64, D1=256, D2=256
    feature_size = F * D1 * D2  # 64 * 256 * 256 = 4194304
    
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    grid = lambda meta: (meta['NUM_CORES'],)
    
    # 启动内核
    aikg_40_LayerNorm_kernel[grid](
        x, y, B, feature_size, 1e-5,
        # BLOCK_SIZE 和 NUM_CORES 由 autotune 自动传入
    )
    
    return y