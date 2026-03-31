import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 512, 'BLOCK_N': 64, 'GROUP_SIZE': 2}),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def aikg_18_Matmul_with_transposed_both_kernel(
    # 指针参数
    a_ptr,
    b_ptr,
    c_ptr,
    # 矩阵维度
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    # 步长参数
    stride_ak, stride_am,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    # 分块参数
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_cores: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """
    矩阵乘法内核：计算 C = A.T @ B.T
    A: [K, M], B: [N, K], C: [M, N]
    """
    
    # 获取当前核心ID
    pid = tl.program_id(0)
    
    # 计算总块数
    num_blocks_m = tl.cdiv(M, BLOCK_M)
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    total_blocks = num_blocks_m * num_blocks_n
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, total_blocks, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // num_blocks_n
        block_n = block_idx % num_blocks_n
        
        # Swizzle2D分组重排优化
        if M >= N:
            # 行优先分组
            task_m_idx, task_n_idx = tl.swizzle2d(block_m, block_n, num_blocks_m, num_blocks_n, GROUP_SIZE)
        else:
            # 列优先分组
            size_gj = GROUP_SIZE * num_blocks_m
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = min(num_blocks_n - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g
        
        # 计算当前块的起始位置
        start_m = task_m_idx * BLOCK_M
        start_n = task_n_idx * BLOCK_N
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k_outer in range(0, K, BLOCK_K):
            start_k = k_outer
            
            # 加载B的块 (B[N, K])
            b_offsets_n = start_n + tl.arange(0, BLOCK_N)
            b_offsets_k = start_k + tl.arange(0, BLOCK_K)
            b_mask = (b_offsets_n[:, None] < N) & (b_offsets_k[None, :] < K)
            b_ptr_base = b_ptr + b_offsets_n[:, None] * stride_bn + b_offsets_k[None, :] * stride_bk
            b_tile = tl.load(b_ptr_base, mask=b_mask, other=0.0)
            
            # 加载A的块 (A[K, M])
            a_offsets_k = start_k + tl.arange(0, BLOCK_K)
            a_offsets_m = start_m + tl.arange(0, BLOCK_M)
            a_mask = (a_offsets_k[:, None] < K) & (a_offsets_m[None, :] < M)
            a_ptr_base = a_ptr + a_offsets_k[:, None] * stride_ak + a_offsets_m[None, :] * stride_am
            a_tile = tl.load(a_ptr_base, mask=a_mask, other=0.0)
            
            # 矩阵乘法: B_tile @ A_tile (转置后相当于 A.T @ B.T)
            # b_tile: [BLOCK_N, BLOCK_K], a_tile: [BLOCK_K, BLOCK_M]
            # 结果: [BLOCK_N, BLOCK_M]，需要转置为 [BLOCK_M, BLOCK_N]
            gemm_result = tl.dot(b_tile, a_tile)
            accumulator += tl.trans(gemm_result)
        
        # 存储结果到C
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
        c_ptr_base = c_ptr + c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr_base, accumulator, mask=c_mask)


def aikg_18_Matmul_with_transposed_both_triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton实现：计算 C = A.T @ B.T
    
    Args:
        A: 输入张量，形状 [K, M]
        B: 输入张量，形状 [N, K]
        
    Returns:
        输出张量，形状 [M, N]
    """
    # 确保输入张量在设备上且为连续
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取矩阵维度
    K, M = A.shape  # A: [K, M]
    N, K2 = B.shape  # B: [N, K]
    assert K == K2, f"矩阵维度不匹配: K={K}, K2={K2}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置核心数
    num_cores = 20  # Ascend 910B4有20个AI Core
    
    # 定义启动网格
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_18_Matmul_with_transposed_both_kernel[grid](
        A, B, C,
        M, K, N,
        A.stride(0), A.stride(1),  # stride_ak, stride_am
        B.stride(0), B.stride(1),  # stride_bn, stride_bk  
        C.stride(0), C.stride(1),  # stride_cm, stride_cn
        num_cores=num_cores,
    )
    
    return C