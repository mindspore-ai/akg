import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32}),
    ],
    key=['B', 'M', 'N'],
)
@triton.jit
def aikg_47_Sum_reduction_over_a_dimension_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_CORES: tl.constexpr = 20,
):
    """
    Sum reduction over dimension 1 (M dimension) with keepdim=True
    Input: X[B, M, N]
    Output: Y[B, 1, N]
    """
    pid = tl.program_id(0)  # 核心ID: 0~NUM_CORES-1
    
    # 计算N维度的块数量
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = B * num_blocks_n
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, NUM_CORES):
        # 计算当前块的batch和n_outer索引
        b_idx = block_idx // num_blocks_n
        n_outer = block_idx % num_blocks_n
        
        # 计算当前N块的起始位置
        n_start = n_outer * BLOCK_SIZE_N
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N
        
        # 初始化累加器 - 使用固定大小的块
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        # 在M维度进行分块处理
        for m_outer in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
            # 计算当前M块的起始位置
            m_start = m_outer * BLOCK_SIZE_M
            m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
            m_mask = m_offsets < M
            
            # 创建2D掩码
            full_mask = m_mask[:, None] & n_mask[None, :]
            
            # 计算内存偏移
            x_offset = b_idx * M * N + m_offsets[:, None] * N + n_offsets[None, :]
            
            # 加载数据
            x_block = tl.load(x_ptr + x_offset, mask=full_mask, other=0.0)
            
            # 累加到accumulator - 使用固定大小的块，通过mask控制有效数据
            accumulator += x_block
        
        # 对accumulator进行最终归约（沿M维度）
        final_result = tl.sum(accumulator, axis=0)
        
        # 存储结果
        y_offset = b_idx * N + n_offsets
        tl.store(y_ptr + y_offset, final_result, mask=n_mask)


def aikg_47_Sum_reduction_over_a_dimension_triton_ascend_torch(x: torch.Tensor):
    """
    Triton implementation of sum reduction over dimension 1
    
    Args:
        x (torch.Tensor): Input tensor of shape (B, M, N)  # batch_size=16, dim1=256, dim2=256
        
    Returns:
        torch.Tensor: Output tensor of shape (B, 1, N)
    """
    # 获取输入形状
    B, M, N = x.shape
    
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 分配输出张量 (B, 1, N)
    y = torch.empty((B, 1, N), dtype=x.dtype, device=x.device)
    
    # 计算网格大小 - 使用固定核心数
    grid = (20,)  # NUM_CORES = 20
    
    # 启动内核
    aikg_47_Sum_reduction_over_a_dimension_kernel[grid](
        x, y, B, M, N
        # BLOCK_SIZE_N 和 BLOCK_SIZE_M 由 autotune 自动传入
    )
    
    return y