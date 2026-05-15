import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'PARALLEL_NUM': 20}),
        triton.Config({'BLOCK_SIZE': 128, 'PARALLEL_NUM': 20}),
        triton.Config({'BLOCK_SIZE': 512, 'PARALLEL_NUM': 20}),
        triton.Config({'BLOCK_SIZE': 256, 'PARALLEL_NUM': 40}),
        triton.Config({'BLOCK_SIZE': 128, 'PARALLEL_NUM': 40}),
    ],
    key=['M', 'N'],
)
@triton.jit
def aikg_52_Argmin_over_a_dimension_kernel(
    x_ptr,
    y_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_bx: tl.constexpr,
    stride_mx: tl.constexpr,
    stride_nx: tl.constexpr,
    stride_by: tl.constexpr,
    stride_ny: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PARALLEL_NUM: tl.constexpr,
):
    """
    Argmin 内核实现，沿指定维度查找最小值索引
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算每个核心处理的块数
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE)
    total_blocks = B * num_blocks_n
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, PARALLEL_NUM):
        # 计算当前块的2D索引
        b = block_idx // num_blocks_n
        j_outer = block_idx % num_blocks_n
        
        # 计算当前块的起始位置
        j_start = j_outer * BLOCK_SIZE
        j_end = min(j_start + BLOCK_SIZE, N)
        actual_block_size = j_end - j_start
        
        # 初始化最小值和索引缓冲区
        min_val_buf = tl.full((BLOCK_SIZE,), float('inf'), dtype=tl.float32)
        min_idx_buf = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        
        # 遍历M维度
        for i in range(M):
            # 计算当前行的偏移量
            row_offset = b * stride_bx + i * stride_mx
            
            # 创建索引掩码
            j_indices = tl.arange(0, BLOCK_SIZE)
            offsets = row_offset + (j_start + j_indices) * stride_nx
            mask = j_indices < actual_block_size
            
            # 加载当前数据块
            x_tile = tl.load(x_ptr + offsets, mask=mask, other=float('inf'))
            
            # 比较并更新最小值和索引
            is_smaller = x_tile < min_val_buf
            
            # 更新最小值
            min_val_buf = tl.where(is_smaller, x_tile, min_val_buf)
            
            # 更新索引
            current_idx = tl.full((BLOCK_SIZE,), i, dtype=tl.int32)
            min_idx_buf = tl.where(is_smaller, current_idx, min_idx_buf)
        
        # 存储结果
        y_offset = b * stride_by + j_start * stride_ny
        y_indices = tl.arange(0, BLOCK_SIZE)
        y_offsets = y_offset + y_indices * stride_ny
        y_mask = y_indices < actual_block_size
        
        tl.store(y_ptr + y_offsets, min_idx_buf, mask=y_mask)


def aikg_52_Argmin_over_a_dimension_triton_ascend_torch(x, dim=1):
    """
    Argmin Triton 实现启动函数
    
    Args:
        x: 输入张量，形状为 [B, M, N]
        dim: 沿哪个维度查找最小值索引
        
    Returns:
        最小值索引张量，形状为 [B, N]
    """
    # 确保输入是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入形状
    B, M, N = x.shape  # B=16, M=256, N=256
    
    # 创建输出张量
    output_shape = list(x.shape)
    output_shape.pop(dim)
    y = torch.empty(output_shape, dtype=torch.int32, device=x.device)
    
    # 计算步长
    stride_bx = x.stride(0)
    stride_mx = x.stride(1)
    stride_nx = x.stride(2)
    stride_by = y.stride(0)
    stride_ny = y.stride(1)
    
    # 使用lambda函数设置grid
    grid = lambda meta: (meta['PARALLEL_NUM'],)
    
    # 启动内核
    aikg_52_Argmin_over_a_dimension_kernel[grid](
        x, y, B, M, N,
        stride_bx, stride_mx, stride_nx,
        stride_by, stride_ny,
        # BLOCK_SIZE和PARALLEL_NUM由autotune自动传入
    )
    
    return y