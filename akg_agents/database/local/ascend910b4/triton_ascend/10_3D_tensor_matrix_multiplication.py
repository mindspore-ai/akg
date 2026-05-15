import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}),
    ],
    key=['M', 'L', 'K'],
)
@triton.jit
def aikg_10_3D_tensor_matrix_multiplication_kernel(
    A_ptr, B_ptr, C_ptr,
    N, M, K, L,
    stride_An, stride_Am, stride_Ak,
    stride_Bk, stride_Bn,
    stride_Cn, stride_Cm, stride_Cl,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_cores: tl.constexpr,
):
    """
    3D张量矩阵乘法内核
    A: [N, M, K] f16
    B: [K, L] f16  
    C: [N, M, L] f32
    """
    # 获取核心ID
    pid = tl.program_id(0)
    
    # 计算总块数
    M_blocks = tl.cdiv(M, BLOCK_M)
    N_blocks = tl.cdiv(L, BLOCK_N)
    total_blocks = N * M_blocks * N_blocks
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, num_cores):
        # 将一维索引转换为三维索引
        n = block_idx // (M_blocks * N_blocks)
        remainder = block_idx % (M_blocks * N_blocks)
        m_block = remainder // N_blocks
        n_block = remainder % N_blocks
        
        # 计算当前块的起始位置
        m_start = m_block * BLOCK_M
        n_start = n_block * BLOCK_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k_outer in range(0, K, BLOCK_K):
            k_start = k_outer
            
            # 加载A的当前块 [BLOCK_M, BLOCK_K]
            a_offsets_m = m_start + tl.arange(0, BLOCK_M)
            a_offsets_k = k_start + tl.arange(0, BLOCK_K)
            
            a_mask = (a_offsets_m[:, None] < M) & (a_offsets_k[None, :] < K)
            a_offset = (n * stride_An + 
                       a_offsets_m[:, None] * stride_Am + 
                       a_offsets_k[None, :] * stride_Ak)
            a_tile = tl.load(A_ptr + a_offset, mask=a_mask, other=0.0)
            
            # 加载B的当前块 [BLOCK_K, BLOCK_N]  
            b_offsets_k = k_start + tl.arange(0, BLOCK_K)
            b_offsets_n = n_start + tl.arange(0, BLOCK_N)
            
            b_mask = (b_offsets_k[:, None] < K) & (b_offsets_n[None, :] < L)
            b_offset = (b_offsets_k[:, None] * stride_Bk + 
                       b_offsets_n[None, :] * stride_Bn)
            b_tile = tl.load(B_ptr + b_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘法累加
            accumulator += tl.dot(a_tile, b_tile)
        
        # 存储结果到C [BLOCK_M, BLOCK_N]
        c_offsets_m = m_start + tl.arange(0, BLOCK_M)
        c_offsets_n = n_start + tl.arange(0, BLOCK_N)
        
        c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < L)
        c_offset = (n * stride_Cn + 
                   c_offsets_m[:, None] * stride_Cm + 
                   c_offsets_n[None, :] * stride_Cl)
        tl.store(C_ptr + c_offset, accumulator, mask=c_mask)


def aikg_10_3D_tensor_matrix_multiplication_triton_ascend_torch(A, B):
    """
    3D张量矩阵乘法启动函数
    A: [N, M, K] f16
    B: [K, L] f16
    返回: [N, M, L] f32
    """
    # 获取输入形状
    N, M, K = A.shape  # N=16, M=1024, K=2048
    K2, L = B.shape    # K2=2048, L=768
    assert K == K2, f"K维度不匹配: {K} != {K2}"
    
    # 确保输入连续
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量
    C = torch.empty((N, M, L), dtype=torch.float32, device=A.device)
    
    # 获取步长信息
    stride_An, stride_Am, stride_Ak = A.stride()
    stride_Bk, stride_Bn = B.stride()
    stride_Cn, stride_Cm, stride_Cl = C.stride()
    
    # 固定核心数启动
    num_cores = 20
    
    # 使用lambda函数设置grid
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_10_3D_tensor_matrix_multiplication_kernel[grid](
        A, B, C,
        N, M, K, L,
        stride_An, stride_Am, stride_Ak,
        stride_Bk, stride_Bn,
        stride_Cn, stride_Cm, stride_Cl,
        num_cores=num_cores,
    )
    
    return C