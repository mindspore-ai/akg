import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}),
        triton.Config({'BLOCK_SIZE_M': 512, 'BLOCK_SIZE_N': 32}),
    ],
    key=['M', 'N'],
)
@triton.jit
def aikg_50_Product_reduction_over_a_dimension_kernel(
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
    Triton 乘积归约内核
    在M维度上进行乘积归约，输出形状为[B, N]
    """
    # 获取程序ID
    pid_b = tl.program_id(0)  # 批次维度
    pid_n = tl.program_id(1)  # N维度
    
    # 计算当前块处理的N范围
    n_start = pid_n * BLOCK_SIZE_N
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < N
    
    # 初始化累加器（初始值为1，因为要做乘积）
    accumulator = tl.full((BLOCK_SIZE_N,), 1.0, dtype=tl.float32)
    
    # 在M维度上进行分块归约
    for m_start in range(0, M, BLOCK_SIZE_M):
        # 计算当前块处理的M范围
        m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_offsets < M
        
        # 计算输入指针偏移
        x_offset = pid_b * stride_xb + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
        
        # 加载数据到UB
        x_buffer = tl.load(x_ptr + x_offset, mask=m_mask[:, None] & n_mask[None, :], other=1.0)
        
        # 计算当前块的乘积并累加到accumulator
        # 使用向量化操作计算每个n位置的M维度乘积
        block_product = tl.full((BLOCK_SIZE_N,), 1.0, dtype=tl.float32)
        
        # 对M维度进行归约
        for m_local in range(BLOCK_SIZE_M):
            if m_start + m_local < M:
                # 提取当前m行的数据
                row_data = tl.extract_slice(x_buffer, (m_local, 0), (1, BLOCK_SIZE_N), (1, 1))
                row_data = tl.reshape(row_data, (BLOCK_SIZE_N,))
                
                # 更新块乘积
                block_product = block_product * row_data
        
        # 累加到最终结果
        accumulator = accumulator * block_product
    
    # 计算输出指针偏移
    y_offset = pid_b * stride_yb + n_offsets * stride_yn
    
    # 存储结果
    tl.store(y_ptr + y_offset, accumulator, mask=n_mask)


def aikg_50_Product_reduction_over_a_dimension_triton_ascend_torch(x: torch.Tensor):
    """
    Triton 乘积归约启动函数
    
    Args:
        x (torch.Tensor): 输入张量，形状为[B, M, N]  # B=16, M=256, N=256
        
    Returns:
        torch.Tensor: 输出张量，形状为[B, N]
    """
    # 获取输入形状参数
    B, M, N = x.shape
    
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量
    y = torch.empty((B, N), dtype=x.dtype, device=x.device)
    
    # 计算网格大小
    grid = lambda meta: (B, triton.cdiv(N, meta['BLOCK_SIZE_N']))
    
    # 启动内核
    aikg_50_Product_reduction_over_a_dimension_kernel[grid](
        x, y, B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1),
    )
    
    return y