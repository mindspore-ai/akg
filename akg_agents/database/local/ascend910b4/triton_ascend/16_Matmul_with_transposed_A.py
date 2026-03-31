import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_16_Matmul_with_transposed_A_kernel(
    # 指针参数
    a_ptr,
    b_ptr,
    c_ptr,
    # 矩阵维度
    M,  # A的列数，C的行数
    K,  # A的行数，B的行数
    N,  # B的列数，C的列数
    # 步长参数
    stride_ak, stride_am,  # A的步长 [K, M]
    stride_bk, stride_bn,  # B的步长 [K, N]
    stride_cm, stride_cn,  # C的步长 [M, N]
    # 分块参数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # 核心数
    num_cores: tl.constexpr,
):
    """
    A转置矩阵乘法内核：C = A^T @ B
    A: [K, M] -> 转置后为 [M, K]
    B: [K, N]
    C: [M, N]
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
        
        # 计算当前块的起始位置
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k_outer in range(0, K, BLOCK_K):
            # 计算当前K块的结束位置
            k_end = min(k_outer + BLOCK_K, K)
            
            # 使用块指针加载A块 (转置访问: A[k, m])
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(K, M),  # A的原始形状
                strides=(stride_ak, stride_am),
                offsets=(k_outer, start_m),
                block_shape=(BLOCK_K, BLOCK_M),
                order=(1, 0)  # 行主序
            )
            
            # 加载A数据 [BLOCK_K, BLOCK_M]
            a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
            
            # 使用块指针加载B块
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(K, N),  # B的原始形状
                strides=(stride_bk, stride_bn),
                offsets=(k_outer, start_n),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)  # 行主序
            )
            
            # 加载B数据 [BLOCK_K, BLOCK_N]
            b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
            
            # 矩阵乘法累加 (A转置后与B相乘)
            # a_tile形状 [BLOCK_K, BLOCK_M] 需要转置为 [BLOCK_M, BLOCK_K]
            # b_tile形状 [BLOCK_K, BLOCK_N]
            a_tile_transposed = tl.trans(a_tile)  # 转置为 [BLOCK_M, BLOCK_K]
            accumulator += tl.dot(a_tile_transposed, b_tile)
        
        # 使用块指针存储结果到C
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),  # C的形状
            strides=(stride_cm, stride_cn),
            offsets=(start_m, start_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)  # 行主序
        )
        
        # 存储数据
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


def aikg_16_Matmul_with_transposed_A_triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton实现：C = A^T @ B
    A: [K, M]
    B: [K, N]
    C: [M, N]
    """
    # 获取输入张量形状
    K, M = A.shape  # A的形状为[K, M]
    K2, N = B.shape  # B的形状为[K, N]
    assert K == K2, f"矩阵维度不匹配: A的K维度{K} != B的K维度{K2}"
    
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 固定核心数
    num_cores = 20
    
    # 使用lambda函数计算grid
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_16_Matmul_with_transposed_A_kernel[grid](
        A, B, C,
        M, K, N,
        A.stride(0), A.stride(1),  # A的步长 [stride_k, stride_m]
        B.stride(0), B.stride(1),  # B的步长 [stride_k, stride_n]
        C.stride(0), C.stride(1),  # C的步长 [stride_m, stride_n]
        num_cores=num_cores,
    )
    
    return C