import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}),
    ],
    key=['M', 'N'],
)
@triton.jit
def aikg_49_Max_reduction_over_a_dimension_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Max reduction over dimension 1 (M dimension) of a 3D tensor [B, M, N]
    """
    # 获取程序ID
    pid_b = tl.program_id(0)  # batch维度
    pid_n = tl.program_id(1)  # N维度分块
    
    # 计算当前处理的N范围
    n_start = pid_n * BLOCK_SIZE_N
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # 初始化累加器为负无穷大
    max_acc = tl.full((BLOCK_SIZE_N,), -float('inf'), dtype=tl.float32)
    
    # 遍历M维度
    for m_start in range(0, M, BLOCK_SIZE_M):
        # 计算当前处理的M范围
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offsets < M
        
        # 创建完整的mask
        full_mask = m_mask[:, None] & n_mask[None, :]
        
        # 计算内存偏移
        x_offset = (pid_b * stride_xb + 
                   m_offsets[:, None] * stride_xm + 
                   n_offsets[None, :] * stride_xn)
        
        # 加载数据块
        x_tile = tl.load(x_ptr + x_offset, mask=full_mask, other=-float('inf'))
        
        # 计算当前块每列的最大值
        current_max = tl.max(x_tile, axis=0)
        
        # 更新全局最大值
        max_acc = tl.maximum(max_acc, current_max)
    
    # 计算输出偏移并存储结果
    y_offset = pid_b * stride_yb + n_offsets * stride_yn
    tl.store(y_ptr + y_offset, max_acc, mask=n_mask)


def aikg_49_Max_reduction_over_a_dimension_triton_ascend_torch(x: torch.Tensor):
    """
    Triton实现：在维度1上进行Max归约
    
    Args:
        x: 输入张量 [B, M, N]
        
    Returns:
        输出张量 [B, N]
    """
    # 获取输入形状
    B, M, N = x.shape  # batch_size=16, M=256, N=256
    
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 创建输出张量 [B, N]
    y = torch.empty((B, N), dtype=x.dtype, device=x.device)
    
    # 计算网格大小
    grid = (B, triton.cdiv(N, 32))  # 使用默认的BLOCK_SIZE_N=32计算初始网格
    
    # 启动内核
    aikg_49_Max_reduction_over_a_dimension_kernel[grid](
        x, y, B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1)
    )
    
    return y