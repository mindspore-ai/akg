import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'TILE_SIZE_N': 4096}),
        triton.Config({'TILE_SIZE_N': 8192}),
        triton.Config({'TILE_SIZE_N': 16384}),
    ],
    key=['N'],
)
@triton.jit
def aikg_91_cumsum_reverse_kernel(
    x_ptr,  # 输入张量指针
    y_ptr,  # 输出张量指针
    B,      # batch size
    N,      # 序列长度
    stride_b,  # batch维度步长
    stride_n,  # 序列维度步长
    TILE_SIZE_N: tl.constexpr,  # 分块大小
):
    """
    反向累积求和内核
    每个程序处理一个batch中的一个分块
    """
    # 获取程序ID
    pid_b = tl.program_id(0)  # batch维度
    pid_n = tl.program_id(1)  # 序列分块维度
    
    # 计算当前分块的起始和结束位置
    n_start = pid_n * TILE_SIZE_N
    n_end = min(n_start + TILE_SIZE_N, N)
    
    # 计算当前batch的偏移
    batch_offset = pid_b * stride_b
    
    # 初始化累加器（标量变量）
    temp_sum = 0.0
    
    # 反向累积求和：从分块末尾向前处理
    for i in range(TILE_SIZE_N - 1, -1, -1):
        n_idx = n_start + i
        if n_idx < n_end:
            # 加载输入值
            offset = batch_offset + n_idx * stride_n
            x_val = tl.load(x_ptr + offset)
            
            # 累加
            temp_sum += x_val
            
            # 存储结果（使用标量值）
            tl.store(y_ptr + offset, temp_sum)


def aikg_91_cumsum_reverse_triton_ascend_torch(x, dim=1):
    """
    Triton反向累积求和启动函数
    
    参数:
        x: 输入张量，形状为 [B, N] (B=128, N=4000)
        dim: 累积求和的维度
    
    返回:
        反向累积求和结果
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状
    B, N = x.shape  # B=128, N=4000
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算步长
    stride_b = x.stride(0)
    stride_n = x.stride(1)
    
    # 计算网格大小
    num_blocks_n = triton.cdiv(N, 2048)  # 使用最小分块大小计算最大网格
    
    # 启动内核
    grid = (B, num_blocks_n)
    
    aikg_91_cumsum_reverse_kernel[grid](
        x, y, B, N, stride_b, stride_n
    )
    
    return y