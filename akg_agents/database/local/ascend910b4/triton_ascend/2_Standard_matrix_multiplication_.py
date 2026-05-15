import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_2_Standard_matrix_multiplication__kernel(
    # 指针参数
    a_ptr,
    b_ptr,
    c_ptr,
    # 矩阵维度
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    # 步长参数
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # 调优参数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_cores: tl.constexpr = 20,
):
    """
    Triton矩阵乘法内核，使用固定核心数启动
    """
    # 获取当前核心ID
    pid = tl.program_id(0)
    
    # 计算总块数
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N
        
        # 计算当前块的起始位置
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k in range(0, K, BLOCK_K):
            # 计算当前K块的起始位置
            start_k = k
            
            # 加载A块 (M×K)
            a_offsets_m = start_m + tl.arange(0, BLOCK_M)
            a_offsets_k = start_k + tl.arange(0, BLOCK_K)
            a_mask = (a_offsets_m[:, None] < M) & (a_offsets_k[None, :] < K)
            a_ptr_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a_tile = tl.load(a_ptr + a_ptr_offset, mask=a_mask, other=0.0)
            
            # 加载B块 (K×N)
            b_offsets_k = start_k + tl.arange(0, BLOCK_K)
            b_offsets_n = start_n + tl.arange(0, BLOCK_N)
            b_mask = (b_offsets_k[:, None] < K) & (b_offsets_n[None, :] < N)
            b_ptr_offset = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
            b_tile = tl.load(b_ptr + b_ptr_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘法累加
            accumulator += tl.dot(a_tile, b_tile)
        
        # 存储结果
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
        c_ptr_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr + c_ptr_offset, accumulator, mask=c_mask)


def aikg_2_Standard_matrix_multiplication__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵乘法启动函数
    
    Args:
        A: 输入张量，形状为 (M, K)
        B: 输入张量，形状为 (K, N)
        
    Returns:
        输出张量，形状为 (M, N)
    """
    # 检查输入维度
    M, K = A.shape  # M=1024, K=4096
    K2, N = B.shape  # K2=4096, N=2048
    assert K == K2, f"矩阵维度不匹配: {K} != {K2}"
    
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置固定核心数
    num_cores = 20
    
    # 使用lambda函数计算grid，autotune会自动选择最佳配置
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_2_Standard_matrix_multiplication__kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_cores=num_cores,
    )
    
    return C