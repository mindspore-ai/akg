import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_N': 256}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512, 'BLOCK_N': 64}),
        triton.Config({'BLOCK_M': 512, 'BLOCK_K': 64, 'BLOCK_N': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_13_Matmul_for_symmetric_matrices_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_CORES: tl.constexpr = 20,
):
    """
    Triton 矩阵乘法内核，用于对称矩阵乘法
    """
    pid = tl.program_id(0)  # 核心ID: 0~NUM_CORES-1
    
    # 计算总块数
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, NUM_CORES):
        # 计算当前块的2D索引
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k in range(0, K, BLOCK_K):
            # 加载A块 (M x K)
            a_offset_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
            a_offset_k = k + tl.arange(0, BLOCK_K)
            a_mask = (a_offset_m[:, None] < M) & (a_offset_k[None, :] < K)
            a = tl.load(
                a_ptr + a_offset_m[:, None] * stride_am + a_offset_k[None, :] * stride_ak,
                mask=a_mask,
                other=0.0
            )
            
            # 加载B块 (K x N)
            b_offset_k = k + tl.arange(0, BLOCK_K)
            b_offset_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            b_mask = (b_offset_k[:, None] < K) & (b_offset_n[None, :] < N)
            b = tl.load(
                b_ptr + b_offset_k[:, None] * stride_bk + b_offset_n[None, :] * stride_bn,
                mask=b_mask,
                other=0.0
            )
            
            # 矩阵乘法累加
            accumulator += tl.dot(a, b)
        
        # 存储结果
        c_offset_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c_offset_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_mask = (c_offset_m[:, None] < M) & (c_offset_n[None, :] < N)
        tl.store(
            c_ptr + c_offset_m[:, None] * stride_cm + c_offset_n[None, :] * stride_cn,
            accumulator,
            mask=c_mask
        )


def aikg_13_Matmul_for_symmetric_matrices_triton_ascend_torch(A, B):
    """
    Triton 矩阵乘法启动函数，用于对称矩阵乘法
    
    Args:
        A (torch.Tensor): 输入矩阵A，形状(N, N)，对称
        B (torch.Tensor): 输入矩阵B，形状(N, N)，对称
        
    Returns:
        torch.Tensor: 输出矩阵C，形状(N, N)
    """
    # 确保输入是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取矩阵维度
    M, K = A.shape  # M = N, K = N
    K2, N = B.shape  # K2 = N, N = N
    assert K == K2, f"矩阵维度不匹配: {K} != {K2}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # 设置固定核心数
    NUM_CORES = 20
    
    # 使用lambda函数设置grid，autotune会自动选择最佳配置
    grid = lambda meta: (NUM_CORES,)
    
    # 启动内核
    aikg_13_Matmul_for_symmetric_matrices_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_CORES=NUM_CORES,
    )
    
    return C