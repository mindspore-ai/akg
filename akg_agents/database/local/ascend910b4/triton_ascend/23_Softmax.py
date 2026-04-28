import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 8192, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_N': 4096, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_N': 2048, 'NUM_CORES': 20}),
        triton.Config({'BLOCK_SIZE_N': 1024, 'NUM_CORES': 40}),
    ],
    key=['B', 'N'],
)
@triton.jit
def aikg_23_Softmax_kernel(
    X_ptr,  # 输入张量指针
    Y_ptr,  # 输出张量指针
    B: tl.constexpr,  # batch_size = 16
    N: tl.constexpr,  # dim = 16384
    stride_xb: tl.constexpr,  # X的batch维度步长
    stride_xn: tl.constexpr,  # X的特征维度步长
    stride_yb: tl.constexpr,  # Y的batch维度步长
    stride_yn: tl.constexpr,  # Y的特征维度步长
    BLOCK_SIZE_N: tl.constexpr,  # 特征维度块大小
    NUM_CORES: tl.constexpr,  # 核心数
):
    """
    Softmax Triton内核实现
    每个核心处理多个batch行
    """
    # 获取核心ID
    pid = tl.program_id(0)
    
    # 计算每个核心处理的batch行数
    rows_per_core = tl.cdiv(B, NUM_CORES)
    core_start = pid * rows_per_core
    core_end = min(core_start + rows_per_core, B)
    
    # 每个核心循环处理分配的batch行
    for b in range(core_start, core_end):
        # 阶段1: 计算最大值
        max_val = -float('inf')
        
        # 分块处理特征维度
        for offset_n in range(0, N, BLOCK_SIZE_N):
            # 计算当前块的偏移和掩码
            n_offsets = offset_n + tl.arange(0, BLOCK_SIZE_N)
            mask = n_offsets < N
            
            # 加载数据块
            x_ptr = X_ptr + b * stride_xb + n_offsets * stride_xn
            x_chunk = tl.load(x_ptr, mask=mask, other=-float('inf'))
            
            # 更新最大值
            chunk_max = tl.max(x_chunk, axis=0)
            max_val = tl.maximum(max_val, chunk_max)
        
        # 阶段2: 计算sum(exp(x - max))
        sum_val = 0.0  # 使用float32标量累加器提高精度
        
        for offset_n in range(0, N, BLOCK_SIZE_N):
            # 计算当前块的偏移和掩码
            n_offsets = offset_n + tl.arange(0, BLOCK_SIZE_N)
            mask = n_offsets < N
            
            # 加载数据块
            x_ptr = X_ptr + b * stride_xb + n_offsets * stride_xn
            x_chunk = tl.load(x_ptr, mask=mask, other=0.0)
            
            # 计算exp(x - max_val)，使用exp提高精度
            x_centered = x_chunk - max_val
            exp_chunk = tl.math.exp(x_centered)
            
            # 累加sum，使用高精度累加
            chunk_sum = tl.sum(exp_chunk, axis=0)
            sum_val += chunk_sum.to(tl.float32)  # 确保高精度累加
        
        # 阶段3: 计算softmax并存储结果
        for offset_n in range(0, N, BLOCK_SIZE_N):
            # 计算当前块的偏移和掩码
            n_offsets = offset_n + tl.arange(0, BLOCK_SIZE_N)
            mask = n_offsets < N
            
            # 加载数据块
            x_ptr = X_ptr + b * stride_xb + n_offsets * stride_xn
            x_chunk = tl.load(x_ptr, mask=mask, other=0.0)
            
            # 计算softmax: exp(x - max_val) / sum_val
            x_centered = x_chunk - max_val
            exp_chunk = tl.math.exp(x_centered)
            result_chunk = exp_chunk / sum_val
            
            # 存储结果
            y_ptr = Y_ptr + b * stride_yb + n_offsets * stride_yn
            tl.store(y_ptr, result_chunk, mask=mask)


def aikg_23_Softmax_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Softmax Triton启动函数
    
    Args:
        x (torch.Tensor): 输入张量，形状为 (batch_size, num_features)
        
    Returns:
        torch.Tensor: Softmax输出，形状与输入相同
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状参数
    B, N = x.shape  # B = 16, N = 16384
    
    # 分配输出张量
    y = torch.empty_like(x)
    
    # 计算网格大小
    grid = lambda meta: (meta['NUM_CORES'],)
    
    # 启动内核
    aikg_23_Softmax_kernel[grid](
        x, y, B, N,
        x.stride(0), x.stride(1),  # X的步长
        y.stride(0), y.stride(1),  # Y的步长
        # BLOCK_SIZE_N和NUM_CORES由autotune自动传入
    )
    
    return y