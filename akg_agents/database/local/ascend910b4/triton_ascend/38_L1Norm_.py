import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_D': 4096, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_D': 2048, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_D': 8192, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_D': 4096, 'NUM_CORES': 16}),
        triton.Config({'BLOCK_SIZE_D': 4096, 'NUM_CORES': 24}),
    ],
    key=['B', 'D'],
)
@triton.jit
def aikg_38_L1Norm__kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    L1归一化内核实现
    每个核心处理多个批次，分两遍计算：第一遍计算绝对值之和，第二遍进行归一化
    """
    # 获取核心ID
    core_id = tl.program_id(0)
    
    # 计算每个核心需要处理的批次范围
    batches_per_core = triton.cdiv(B, NUM_CORES)
    
    # 第一遍：计算每个批次的绝对值之和
    for batch_idx in range(batches_per_core):
        b = core_id * batches_per_core + batch_idx
        if b < B:
            # 初始化当前批次的绝对值之和（使用标量变量）
            batch_sum = 0.0
            
            # 处理当前批次的所有数据块
            for d_outer in range(0, D, BLOCK_SIZE_D):
                # 计算偏移量 - 使用固定BLOCK_SIZE_D而非动态block_size
                offsets = b * D + d_outer + tl.arange(0, BLOCK_SIZE_D)
                mask = offsets < b * D + D
                
                # 加载数据
                x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                
                # 计算绝对值
                abs_data = tl.abs(x_data)
                
                # 累加块内和
                block_sum = tl.sum(abs_data, axis=0)
                batch_sum += block_sum
            
            # 第二遍：归一化计算
            for d_outer in range(0, D, BLOCK_SIZE_D):
                # 计算偏移量
                offsets = b * D + d_outer + tl.arange(0, BLOCK_SIZE_D)
                mask = offsets < b * D + D
                
                # 加载数据
                x_data = tl.load(x_ptr + offsets, mask=mask, other=0.0)
                
                # 归一化计算
                y_data = x_data / batch_sum
                
                # 存储结果
                tl.store(y_ptr + offsets, y_data, mask=mask)


def aikg_38_L1Norm__triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    L1归一化Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，形状为(B, D)  # B=16, D=16384
        
    Returns:
        torch.Tensor: 输出张量，形状与输入相同
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状
    B, D = x.shape
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 启动内核
    grid = lambda meta: (meta['NUM_CORES'],)
    
    aikg_38_L1Norm__kernel[grid](
        x, y, B, D,
        # BLOCK_SIZE_D和NUM_CORES由autotune自动传入
    )
    
    return y