import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 256, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 128, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_N': 256}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512, 'BLOCK_N': 64}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 1024, 'BLOCK_N': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_1_Square_matrix_multiplication__kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A的步长: [M, K]
    stride_bk, stride_bn,  # B的步长: [K, N]
    stride_cm, stride_cn,  # C的步长: [M, N]
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton矩阵乘法内核 - 方阵乘法优化版本
    """
    # 计算总块数
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    
    # 获取当前核心ID
    pid = tl.program_id(0)
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k in range(0, K, BLOCK_K):
            # 计算A块的偏移和掩码
            a_offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
            a_offsets_k = k + tl.arange(0, BLOCK_K)
            
            # 创建A块的掩码
            a_mask_m = a_offsets_m < M
            a_mask_k = a_offsets_k < K
            a_mask = a_mask_m[:, None] & a_mask_k[None, :]
            
            # 加载A块
            a_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)
            
            # 计算B块的偏移和掩码
            b_offsets_k = k + tl.arange(0, BLOCK_K)
            b_offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            
            # 创建B块的掩码
            b_mask_k = b_offsets_k < K
            b_mask_n = b_offsets_n < N
            b_mask = b_mask_k[:, None] & b_mask_n[None, :]
            
            # 加载B块
            b_offset = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘累加
            accumulator += tl.dot(a, b)
        
        # 计算C块的偏移和掩码
        c_offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c_offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        # 创建C块的掩码
        c_mask_m = c_offsets_m < M
        c_mask_n = c_offsets_n < N
        c_mask = c_mask_m[:, None] & c_mask_n[None, :]
        
        # 存储结果
        c_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)


def aikg_1_Square_matrix_multiplication__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵乘法启动函数 - 方阵乘法
    
    Args:
        A (torch.Tensor): 输入矩阵A，形状为(N, N)
        B (torch.Tensor): 输入矩阵B，形状为(N, N)
        
    Returns:
        torch.Tensor: 输出矩阵C，形状为(N, N)
    """
    # 确保输入是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取矩阵维度
    M, K = A.shape  # A: (M, K) = (N, N)
    K2, N = B.shape  # B: (K, N) = (N, N)
    assert K == K2, f"矩阵维度不匹配: A的列数{K} != B的行数{K2}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 固定核心数 (Ascend 910B4有20个AI Core)
    num_cores = 20
    
    # 使用lambda函数计算grid，autotune会自动选择最佳配置
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_1_Square_matrix_multiplication__kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),  # A的步长
        B.stride(0), B.stride(1),  # B的步长
        C.stride(0), C.stride(1),  # C的步长
        num_cores=num_cores,
        # BLOCK_M, BLOCK_K, BLOCK_N 由autotune自动传入
    )
    
    return C