import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 1024}),
    ],
    key=['N'],
)
@triton.jit
def aikg_24_LogSoftmax_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    B,      # batch size
    N,      # feature dimension
    dim: tl.constexpr,  # softmax维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """
    LogSoftmax Triton内核实现
    每个程序处理一行数据
    """
    # 获取当前程序处理的batch索引
    b = tl.program_id(0)
    
    # 计算当前行的偏移量
    row_offset = b * N
    
    # 阶段1: 计算最大值
    max_val = -float('inf')
    
    # 分块处理当前行
    for i in range(0, N, BLOCK_SIZE):
        # 计算当前块的偏移和掩码
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < N
        
        # 加载数据块
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
        
        # 计算当前块的最大值
        chunk_max = tl.max(x_chunk, axis=0)
        
        # 更新全局最大值
        max_val = tl.maximum(max_val, chunk_max)
    
    # 阶段2: 计算exp和
    sum_val = 0.0
    
    for i in range(0, N, BLOCK_SIZE):
        # 计算当前块的偏移和掩码
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < N
        
        # 加载数据块
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # 减去最大值并计算exp
        x_stable = x_chunk - max_val
        exp_chunk = tl.math.exp(x_stable)  # 使用标准exp函数
        
        # 累加exp值
        chunk_sum = tl.sum(exp_chunk, axis=0)
        sum_val += chunk_sum
    
    # 阶段3: 计算log(sum)
    log_sum = tl.math.log(sum_val)  # 使用标准log函数
    
    # 阶段4: 计算最终结果
    for i in range(0, N, BLOCK_SIZE):
        # 计算当前块的偏移和掩码
        offsets = row_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = (i + tl.arange(0, BLOCK_SIZE)) < N
        
        # 加载数据块
        x_chunk = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # 计算LogSoftmax: log(exp(x - max) / sum) = (x - max) - log(sum)
        result = (x_chunk - max_val) - log_sum
        
        # 存储结果
        tl.store(y_ptr + offsets, result, mask=mask)


def aikg_24_LogSoftmax_triton_ascend_torch(x: torch.Tensor, dim: int = 1):
    """
    LogSoftmax Triton实现启动函数
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, dim)
        dim (int): softmax维度，默认为1
        
    Returns:
        torch.Tensor: LogSoftmax结果，形状与输入相同
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状
    B, N = x.shape  # B=16, N=16384
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    grid = (B,)
    
    # 启动内核 - 注意：autotune参数不应显式传递
    aikg_24_LogSoftmax_kernel[grid](
        x, y, B, N, dim
        # BLOCK_SIZE参数由autotune自动传入，不应显式指定
    )
    
    return y