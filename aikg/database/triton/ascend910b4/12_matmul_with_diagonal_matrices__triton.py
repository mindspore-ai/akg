import torch
import triton
import triton.language as tl

@triton.jit
def matmul_with_diagonal_matrices__kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    N,
    M,
    stride_B0,  
    stride_B1,  
    stride_C0,  
    stride_C1,  
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    """
    Triton内核实现对角矩阵乘法：C = diag(A) * B
    """
    pid = tl.program_id(0)
    start_row = pid * BLOCK_SIZE_ROW
    
    row_indices = tl.arange(0, BLOCK_SIZE_ROW)
    row_mask = (start_row + row_indices) < N
    
    # 加载A向量（对角线元素）
    A_offsets = start_row + row_indices
    A = tl.load(A_ptr + A_offsets, mask=row_mask, other=0.0)
    
    total_iters = tl.cdiv(M, BLOCK_SIZE_COL)
    
    # 预加载第一个B块
    col_indices0 = tl.arange(0, BLOCK_SIZE_COL)
    col_mask0 = col_indices0 < M  # 初始列掩码
    
    B_ptrs = B_ptr + (start_row + row_indices[:, None]) * stride_B0 + col_indices0[None, :] * stride_B1
    B_current = tl.load(B_ptrs, mask=row_mask[:, None] & col_mask0[None, :], other=0.0)
    
    for j in range(total_iters):
        current_col = j * BLOCK_SIZE_COL
        
        # 计算当前列掩码
        col_indices = tl.arange(0, BLOCK_SIZE_COL)
        col_mask = col_indices < (M - current_col)
        
        # 计算：C_tile = diag(A) * B_current
        C_tile = A[:, None] * B_current
        
        # 存储结果
        C_ptrs = C_ptr + (start_row + row_indices[:, None]) * stride_C0 + (current_col + col_indices[None, :]) * stride_C1
        tl.store(C_ptrs, C_tile, mask=row_mask[:, None] & col_mask[None, :])
        
        # 预加载下一个B块（双缓冲）
        if j < total_iters - 1:
            next_col = (j + 1) * BLOCK_SIZE_COL
            col_mask_next = tl.arange(0, BLOCK_SIZE_COL) < (M - next_col)
            
            B_ptrs_next = B_ptr + (start_row + row_indices[:, None]) * stride_B0 + (next_col + tl.arange(0, BLOCK_SIZE_COL)[None, :]) * stride_B1
            B_next = tl.load(B_ptrs_next, mask=row_mask[:, None] & col_mask_next[None, :], other=0.0)
            B_current = B_next


def matmul_with_diagonal_matrices__triton_torch(A: torch.Tensor, B: torch.Tensor):
    """
    Triton启动函数：对角矩阵乘法
    Args:
        A (torch.Tensor): 一维张量，形状(N,)
        B (torch.Tensor): 二维张量，形状(N, M)
    Returns:
        torch.Tensor: diag(A) @ B 结果，形状(N, M)
    """
    assert A.dim() == 1, "A必须是1维张量"
    assert B.dim() == 2, "B必须是2维张量"
    N = A.shape[0]
    assert B.shape[0] == N, "A和B的第一维必须相同"
    M = B.shape[1]
    
    C = torch.empty((N, M), device=B.device, dtype=B.dtype)
    
    # 使用优化后的块大小（128x32）
    BLOCK_SIZE_ROW = 128
    BLOCK_SIZE_COL = 32
    
    grid = (triton.cdiv(N, BLOCK_SIZE_ROW),)
    
    matmul_with_diagonal_matrices__kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        N=N,
        M=M,
        stride_B0=B.stride(0),
        stride_B1=B.stride(1),
        stride_C0=C.stride(0),
        stride_C1=C.stride(1),
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL
    )
    
    return C