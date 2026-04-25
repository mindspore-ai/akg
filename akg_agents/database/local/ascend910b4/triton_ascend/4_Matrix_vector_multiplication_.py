import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 16}),
    ],
    key=['M', 'K'],
)
@triton.jit
def aikg_4_Matrix_vector_multiplication__kernel(
    A_ptr,  # 输入矩阵A指针
    B_ptr,  # 输入向量B指针  
    C_ptr,  # 输出向量C指针
    M: tl.constexpr,  # 矩阵行数
    K: tl.constexpr,  # 矩阵列数/向量长度
    stride_am: tl.constexpr,  # A的行步长
    stride_ak: tl.constexpr,  # A的列步长
    stride_bk: tl.constexpr,  # B的行步长
    stride_cm: tl.constexpr,  # C的行步长
    BLOCK_M: tl.constexpr,  # M维度块大小
    BLOCK_K: tl.constexpr,  # K维度块大小
):
    """
    矩阵向量乘法内核：C = A @ B
    A: [M, K], B: [K, 1], C: [M, 1]
    """
    # 获取当前程序处理的M范围
    pid_m = tl.program_id(0)
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < M
    
    # 初始化累加器（使用float32防止精度丢失）
    c_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    # K维度循环分块处理
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        
        # 加载A的当前块 [BLOCK_M, BLOCK_K]
        a_offsets = m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_tile = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # 加载B的当前块 [BLOCK_K]
        b_offsets = k_offsets * stride_bk
        b_tile = tl.load(B_ptr + b_offsets, mask=k_mask, other=0.0)
        
        # 广播B到与A相同的形状并计算逐元素乘法
        b_broadcast = tl.broadcast_to(b_tile[None, :], (BLOCK_M, BLOCK_K))
        product = a_tile * b_broadcast
        
        # 沿K维度求和并累加
        k_sum = tl.sum(product, axis=1)
        c_acc += k_sum
    
    # 存储结果到C
    c_offsets = m_offsets * stride_cm
    tl.store(C_ptr + c_offsets, c_acc, mask=m_mask)


def aikg_4_Matrix_vector_multiplication__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵向量乘法启动函数
    
    Args:
        A: 输入矩阵，形状为 (M, K)
        B: 输入向量，形状为 (K, 1)
        
    Returns:
        输出向量，形状为 (M, 1)
    """
    # 确保输入张量在设备上连续
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 获取输入形状
    M, K = A.shape  # M=256, K=131072
    K2, _ = B.shape  # K2=131072, _=1
    assert K == K2, f"矩阵维度不匹配: A的K={K}, B的K={K2}"
    
    # 分配输出张量（使用float32防止精度丢失）
    C = torch.empty((M, 1), dtype=torch.float32, device=A.device)
    
    # 计算网格大小
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    
    # 启动内核
    aikg_4_Matrix_vector_multiplication__kernel[grid](
        A, B, C,
        M, K,
        A.stride(0), A.stride(1),  # stride_am, stride_ak
        B.stride(0),  # stride_bk
        C.stride(0),  # stride_cm
    )
    
    return C