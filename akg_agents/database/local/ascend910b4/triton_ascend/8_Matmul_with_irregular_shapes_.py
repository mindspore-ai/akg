import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 128, 'BLOCK_N': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_8_Matmul_with_irregular_shapes__kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton矩阵乘法内核，支持不规则形状，优化CBUF使用
    """
    pid = tl.program_id(0)  # 核心ID: 0~num_cores-1
    
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
            # 计算当前K块的结束位置
            end_k = min(k + BLOCK_K, K)
            
            # 加载A块 - 使用正确的stride计算
            a_offsets_m = start_m + tl.arange(0, BLOCK_M)
            a_offsets_k = k + tl.arange(0, BLOCK_K)
            
            a_mask_m = a_offsets_m < M
            a_mask_k = a_offsets_k < K
            a_mask = a_mask_m[:, None] & a_mask_k[None, :]
            
            # 正确的stride计算：行主序布局
            a_offset = a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)
            
            # 加载B块 - 使用正确的stride计算
            b_offsets_k = k + tl.arange(0, BLOCK_K)
            b_offsets_n = start_n + tl.arange(0, BLOCK_N)
            
            b_mask_k = b_offsets_k < K
            b_mask_n = b_offsets_n < N
            b_mask = b_mask_k[:, None] & b_mask_n[None, :]
            
            # 正确的stride计算：行主序布局
            b_offset = b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)
            
            # 矩阵乘累加 - 确保数据类型正确
            a_f32 = a.to(tl.float32)
            b_f32 = b.to(tl.float32)
            accumulator += tl.dot(a_f32, b_f32)
        
        # 存储结果
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        
        c_mask_m = c_offsets_m < M
        c_mask_n = c_offsets_n < N
        c_mask = c_mask_m[:, None] & c_mask_n[None, :]
        
        c_offset = c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)


def aikg_8_Matmul_with_irregular_shapes__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵乘法启动函数
    
    Args:
        A: 输入张量，形状为 (M, K)  # M=8205, K=2949
        B: 输入张量，形状为 (K, N)  # K=2949, N=5921
        
    Returns:
        C: 输出张量，形状为 (M, N)
    """
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取输入形状
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"矩阵维度不匹配: {K} != {K2}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置核心数（Ascend 910B4有20个AI Core）
    num_cores = 20
    
    # 使用lambda函数设置grid，autotune会自动选择最佳配置
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_8_Matmul_with_irregular_shapes__kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_cores=num_cores,
    )
    
    return C