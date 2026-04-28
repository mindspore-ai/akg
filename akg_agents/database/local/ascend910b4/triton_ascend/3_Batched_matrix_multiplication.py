import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 256}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 256}),
    ],
    key=['batch_size', 'm', 'k', 'n'],
)
@triton.jit
def aikg_3_Batched_matrix_multiplication_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_size, m, k, n,
    stride_ab, stride_am, stride_ak,  # A的strides: [batch, m, k]
    stride_bb, stride_bk, stride_bn,  # B的strides: [batch, k, n]
    stride_cb, stride_cm, stride_cn,  # C的strides: [batch, m, n]
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched matrix multiplication kernel for Ascend 910B4
    Computes C = A @ B for each batch
    """
    # 计算每个batch的分块数量
    NUM_BLOCKS_M = tl.cdiv(m, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(n, BLOCK_N)
    NUM_BLOCKS_PER_BATCH = NUM_BLOCKS_M * NUM_BLOCKS_N
    TOTAL_BLOCKS = batch_size * NUM_BLOCKS_PER_BATCH
    
    # 获取当前核心ID
    pid = tl.program_id(0)
    
    # 每个核心循环处理多个块
    for block_idx in range(pid, TOTAL_BLOCKS, num_cores):
        # 计算当前块对应的批次和块索引
        batch_idx = block_idx // NUM_BLOCKS_PER_BATCH
        block_in_batch = block_idx % NUM_BLOCKS_PER_BATCH
        
        # 计算当前块的2D索引
        block_m = block_in_batch // NUM_BLOCKS_N
        block_n = block_in_batch % NUM_BLOCKS_N
        
        # 计算当前块的起始位置
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N
        
        # 初始化累加器 (使用L0C)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环 (使用CUBE单元)
        for k_inner in range(0, k, BLOCK_K):
            # 计算当前K块的结束位置
            end_k = min(k_inner + BLOCK_K, k)
            
            # 创建A块指针 - 使用固定大小的BLOCK_K，配合mask处理实际大小
            a_offsets_m = start_m + tl.arange(0, BLOCK_M)
            a_offsets_k = k_inner + tl.arange(0, BLOCK_K)
            a_mask = (a_offsets_m[:, None] < m) & (a_offsets_k[None, :] < k) & (a_offsets_k[None, :] < end_k)
            
            # 加载A块数据 (使用L0A)
            a_ptr_batch = A_ptr + batch_idx * stride_ab
            a_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a_tile = tl.load(a_ptr_batch + a_offset, mask=a_mask, other=0.0)
            
            # 创建B块指针 - 使用固定大小的BLOCK_K，配合mask处理实际大小
            b_offsets_k = k_inner + tl.arange(0, BLOCK_K)
            b_offsets_n = start_n + tl.arange(0, BLOCK_N)
            b_mask = (b_offsets_k[:, None] < k) & (b_offsets_n[None, :] < n) & (b_offsets_k[:, None] < end_k)
            
            # 加载B块数据 (使用L0B)
            b_ptr_batch = B_ptr + batch_idx * stride_bb
            b_offset = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
            b_tile = tl.load(b_ptr_batch + b_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘累加 (使用CUBE单元)
            accumulator += tl.dot(a_tile, b_tile)
        
        # 创建结果指针
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        c_mask = (c_offsets_m[:, None] < m) & (c_offsets_n[None, :] < n)
        
        # 存储结果
        c_ptr_batch = C_ptr + batch_idx * stride_cb
        c_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr_batch + c_offset, accumulator, mask=c_mask)


def aikg_3_Batched_matrix_multiplication_triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton Ascend implementation of batched matrix multiplication
    
    Args:
        A: Input tensor of shape (batch_size, m, k)
        B: Input tensor of shape (batch_size, k, n)
        
    Returns:
        C: Output tensor of shape (batch_size, m, n)
    """
    # 获取输入形状
    batch_size, m, k = A.shape  # batch_size=128, m=128, k=256
    batch_size_b, k_b, n = B.shape  # batch_size_b=128, k_b=256, n=512
    
    # 验证输入形状
    assert batch_size == batch_size_b, f"Batch size mismatch: {batch_size} != {batch_size_b}"
    assert k == k_b, f"K dimension mismatch: {k} != {k_b}"
    
    # 确保输入是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量 (使用float32避免精度问题)
    C = torch.empty((batch_size, m, n), dtype=torch.float32, device=A.device)
    
    # 固定核心数启动 (Ascend 910B4有20个AI Core)
    num_cores = 20
    
    # 使用lambda函数设置grid，autotune会自动选择最佳配置
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_3_Batched_matrix_multiplication_kernel[grid](
        A, B, C,
        batch_size, m, k, n,
        A.stride(0), A.stride(1), A.stride(2),  # A strides: [batch, m, k]
        B.stride(0), B.stride(1), B.stride(2),  # B strides: [batch, k, n]
        C.stride(0), C.stride(1), C.stride(2),  # C strides: [batch, m, n]
        num_cores=num_cores,
        # BLOCK_M, BLOCK_N, BLOCK_K 由autotune自动传入
    )
    
    return C