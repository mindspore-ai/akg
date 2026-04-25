import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 256}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_17_Matmul_with_transposed_B_kernel(
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
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # 分块参数
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Matmul with transposed B kernel
    计算 C = A @ B.T，其中 A 形状为 (M, K)，B 形状为 (N, K)
    """
    # 获取核心ID
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
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环 - 使用BLOCK_K作为constexpr步长
        for k_outer in range(0, K, BLOCK_K):
            # 计算当前K块的起始位置
            k_start = k_outer
            
            # 加载A块 (BLOCK_M × BLOCK_K)
            a_offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
            a_offsets_k = k_start + tl.arange(0, BLOCK_K)
            
            a_mask_m = a_offsets_m[:, None] < M
            a_mask_k = a_offsets_k[None, :] < K
            a_mask = a_mask_m & a_mask_k
            
            a_ptr_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a_tile = tl.load(a_ptr + a_ptr_offset, mask=a_mask, other=0.0)
            
            # 加载B块 (BLOCK_N × BLOCK_K) - 注意B需要转置
            b_offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            b_offsets_k = k_start + tl.arange(0, BLOCK_K)
            
            b_mask_n = b_offsets_n[:, None] < N
            b_mask_k = b_offsets_k[None, :] < K
            b_mask = b_mask_n & b_mask_k
            
            b_ptr_offset = b_offsets_n[:, None] * stride_bn + b_offsets_k[None, :] * stride_bk
            b_tile = tl.load(b_ptr + b_ptr_offset, mask=b_mask, other=0.0)
            
            # 执行矩阵乘法: A @ B.T (使用CUBE单元)
            # 注意：b_tile需要转置为 (BLOCK_K × BLOCK_N) 才能与A相乘
            b_tile_transposed = tl.trans(b_tile)
            accumulator += tl.dot(a_tile, b_tile_transposed)
        
        # 存储结果到全局内存
        c_offsets_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        c_offsets_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
        
        c_mask_m = c_offsets_m[:, None] < M
        c_mask_n = c_offsets_n[None, :] < N
        c_mask = c_mask_m & c_mask_n
        
        c_ptr_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr + c_ptr_offset, accumulator, mask=c_mask)


def aikg_17_Matmul_with_transposed_B_triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton实现矩阵乘法 C = A @ B.T
    
    Args:
        A: 输入张量，形状为 (M, K)  # M=1024, K=4096
        B: 输入张量，形状为 (N, K)  # N=2048, K=4096
        
    Returns:
        输出张量，形状为 (M, N)
    """
    # 获取输入形状
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"矩阵维度不匹配: K={K}, K2={K2}"
    
    # 确保输入张量在设备上且连续
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置核心数
    num_cores = 20
    
    # 启动内核
    grid = lambda meta: (num_cores,)
    
    aikg_17_Matmul_with_transposed_B_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_cores=num_cores,
    )
    
    return C