import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_9_Tall_skinny_matrix_multiplication__kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A的stride
    stride_bk, stride_bn,  # B的stride
    stride_cm, stride_cn,  # C的stride
    NUM_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Tall skinny矩阵乘法内核
    A: [M, K], B: [K, N] -> C: [M, N]
    """
    # 计算总块数
    pid = tl.program_id(0)  # 核心ID: 0~NUM_CORES-1
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n

    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, NUM_CORES):
        # 计算当前块的2D索引
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n
        
        # 计算当前块的起始位置
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k_outer in range(0, K, BLOCK_K):
            start_k = k_outer
            
            # 加载A块 [BLOCK_M, BLOCK_K]
            a_offsets_m = start_m + tl.arange(0, BLOCK_M)
            a_offsets_k = start_k + tl.arange(0, BLOCK_K)
            a_mask = (a_offsets_m[:, None] < M) & (a_offsets_k[None, :] < K)
            a_ptr_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a_tile = tl.load(a_ptr + a_ptr_offset, mask=a_mask, other=0.0)
            
            # 加载B块 [BLOCK_K, BLOCK_N]
            b_offsets_k = start_k + tl.arange(0, BLOCK_K)
            b_offsets_n = start_n + tl.arange(0, BLOCK_N)
            b_mask = (b_offsets_k[:, None] < K) & (b_offsets_n[None, :] < N)
            b_ptr_offset = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
            b_tile = tl.load(b_ptr + b_ptr_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘法累加
            accumulator += tl.dot(a_tile, b_tile)
        
        # 存储结果 [BLOCK_M, BLOCK_N]
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
        c_ptr_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr + c_ptr_offset, accumulator, mask=c_mask)


def aikg_9_Tall_skinny_matrix_multiplication__triton_ascend_torch(A, B):
    """
    Tall skinny矩阵乘法启动函数
    Args:
        A: torch.Tensor of shape (M, K)
        B: torch.Tensor of shape (K, N)
    Returns:
        torch.Tensor of shape (M, N)
    """
    # 确保输入是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取输入形状
    M, K = A.shape  # M=16384, K=16
    K2, N = B.shape  # K2=16, N=16384
    assert K == K2, f"矩阵维度不匹配: A的K={K}, B的K={K2}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置核心数
    NUM_CORES = 20
    
    # 使用lambda函数计算grid
    grid = lambda meta: (NUM_CORES,)
    
    # 启动内核
    aikg_9_Tall_skinny_matrix_multiplication__kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),  # stride_am, stride_ak
        B.stride(0), B.stride(1),  # stride_bk, stride_bn  
        C.stride(0), C.stride(1),  # stride_cm, stride_cn
        NUM_CORES=NUM_CORES,
        # BLOCK_M, BLOCK_N, BLOCK_K 由autotune自动传入
    )
    
    return C