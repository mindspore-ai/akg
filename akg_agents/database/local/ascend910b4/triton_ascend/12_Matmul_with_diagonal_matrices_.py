import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 96}),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 48}),
        triton.Config({'BLOCK_SIZE_N': 96, 'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_N': 48, 'BLOCK_SIZE_M': 256}),
    ],
    key=['N', 'M'],
)
@triton.jit
def aikg_12_Matmul_with_diagonal_matrices__kernel(
    A_ptr,  # 对角矩阵A的指针，形状(N,)
    B_ptr,  # 矩阵B的指针，形状(N, M)
    C_ptr,  # 输出矩阵C的指针，形状(N, M)
    N: tl.constexpr,  # A的长度，B的行数
    M: tl.constexpr,  # B的列数，C的列数
    stride_Bn: tl.constexpr,  # B矩阵n维度的步长
    stride_Bm: tl.constexpr,  # B矩阵m维度的步长
    stride_Cn: tl.constexpr,  # C矩阵n维度的步长
    stride_Cm: tl.constexpr,  # C矩阵m维度的步长
    num_cores: tl.constexpr,  # 核心数
    BLOCK_SIZE_N: tl.constexpr,  # N方向的块大小
    BLOCK_SIZE_M: tl.constexpr,  # M方向的块大小
):
    """
    对角矩阵与矩阵乘法内核
    C = diag(A) * B
    """
    # 获取核心ID
    core_idx = tl.program_id(0)
    
    # 计算每个核心负责的N范围
    n_per_core = tl.cdiv(N, num_cores)
    n_start = core_idx * n_per_core
    n_end = min(n_start + n_per_core, N)
    
    # 如果没有数据需要处理，直接返回
    if n_start >= N:
        return
    
    # 在M方向循环处理
    for m_start in range(0, M, BLOCK_SIZE_M):
        m_end = min(m_start + BLOCK_SIZE_M, M)
        m_size = m_end - m_start
        
        # 在N方向循环处理（避免UB溢出）
        for n_block_start in range(n_start, n_end, BLOCK_SIZE_N):
            n_block_end = min(n_block_start + BLOCK_SIZE_N, n_end)
            n_size = n_block_end - n_block_start
            
            # 加载A的对角元素块到UB
            a_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
            a_mask = a_offsets < n_block_end
            a_tile = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
            
            # 计算B和C的偏移量
            b_offsets_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)[:, None]
            b_offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)[None, :]
            b_mask = (b_offsets_n < n_block_end) & (b_offsets_m < m_end)
            
            c_offsets_n = n_block_start + tl.arange(0, BLOCK_SIZE_N)[:, None]
            c_offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)[None, :]
            c_mask = (c_offsets_n < n_block_end) & (c_offsets_m < m_end)
            
            # 加载B矩阵块
            b_ptr = B_ptr + b_offsets_n * stride_Bn + b_offsets_m * stride_Bm
            b_tile = tl.load(b_ptr, mask=b_mask, other=0.0)
            
            # 执行逐元素乘法: C[i,j] = A[i] * B[i,j]
            # 将a_tile扩展到与b_tile相同的形状
            a_expanded = tl.broadcast_to(a_tile[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))
            c_tile = a_expanded * b_tile
            
            # 存储结果
            c_ptr = C_ptr + c_offsets_n * stride_Cn + c_offsets_m * stride_Cm
            tl.store(c_ptr, c_tile, mask=c_mask)


def aikg_12_Matmul_with_diagonal_matrices__triton_ascend_torch(A, B):
    """
    Triton实现的对角矩阵与矩阵乘法
    
    Args:
        A (torch.Tensor): 1D张量，表示对角矩阵的对角元素，形状(N,)
        B (torch.Tensor): 2D张量，形状(N, M)
        
    Returns:
        torch.Tensor: 计算结果，形状(N, M)
    """
    # 检查输入形状
    N = A.shape[0]  # A的长度，B的行数
    M = B.shape[1]  # B的列数
    assert B.shape[0] == N, f"B的行数({B.shape[0]})必须等于A的长度({N})"
    
    # 分配输出张量
    C = torch.empty((N, M), dtype=A.dtype, device=A.device)
    
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取步长信息
    stride_Bn = B.stride(0)
    stride_Bm = B.stride(1)
    stride_Cn = C.stride(0)
    stride_Cm = C.stride(1)
    
    # 设置核心数
    num_cores = 20  # Ascend 910B4有20个AI Core
    
    # 启动内核
    grid = lambda meta: (num_cores,)
    aikg_12_Matmul_with_diagonal_matrices__kernel[grid](
        A, B, C, N, M, stride_Bn, stride_Bm, stride_Cn, stride_Cm, num_cores
        # BLOCK_SIZE_N和BLOCK_SIZE_M由autotune自动传入
    )
    
    return C