import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def aikg_7_Matmul_with_small_K_dimension__kernel(
    # 指针参数
    a_ptr,
    b_ptr,
    c_ptr,
    # 矩阵维度
    M,  # A的行数，C的行数
    N,  # B的列数，C的列数
    K: tl.constexpr,  # A的列数，B的行数（编译时常量）
    # 步长参数
    stride_am,  # A的行步长
    stride_ak,  # A的列步长
    stride_bk,  # B的行步长
    stride_bn,  # B的列步长
    stride_cm,  # C的行步长
    stride_cn,  # C的列步长
    # 编译时常量
    BLOCK_M: tl.constexpr,  # M方向块大小
    BLOCK_N: tl.constexpr,  # N方向块大小
    num_cores: tl.constexpr = 20,  # AI Core数量
):
    """
    Triton矩阵乘法内核，针对小K维度优化
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
        
        # 计算当前块的起始位置
        start_m = block_m * BLOCK_M
        start_n = block_n * BLOCK_N
        
        # 初始化累加器（L0C）
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # 一次性加载整个K维度（K=32很小）
        # 加载A块到L0A
        a_offsets_m = start_m + tl.arange(0, BLOCK_M)
        a_offsets_k = tl.arange(0, K)  # K现在是编译时常量
        a_mask = (a_offsets_m[:, None] < M) & (a_offsets_k[None, :] < K)
        a_tile = tl.load(
            a_ptr + a_offsets_m[:, None] * stride_am + a_offsets_k[None, :] * stride_ak,
            mask=a_mask,
            other=0.0
        )
        
        # 加载B块到L0B
        b_offsets_k = tl.arange(0, K)  # K现在是编译时常量
        b_offsets_n = start_n + tl.arange(0, BLOCK_N)
        b_mask = (b_offsets_k[:, None] < K) & (b_offsets_n[None, :] < N)
        b_tile = tl.load(
            b_ptr + b_offsets_k[:, None] * stride_bk + b_offsets_n[None, :] * stride_bn,
            mask=b_mask,
            other=0.0
        )
        
        # 矩阵乘法（CUBE单元）
        accumulator += tl.dot(a_tile, b_tile)
        
        # 存储结果到全局内存
        c_offsets_m = start_m + tl.arange(0, BLOCK_M)
        c_offsets_n = start_n + tl.arange(0, BLOCK_N)
        c_mask = (c_offsets_m[:, None] < M) & (c_offsets_n[None, :] < N)
        tl.store(
            c_ptr + c_offsets_m[:, None] * stride_cm + c_offsets_n[None, :] * stride_cn,
            accumulator,
            mask=c_mask
        )


def aikg_7_Matmul_with_small_K_dimension__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵乘法启动函数
    
    Args:
        A: 输入张量，形状为 (M, K)
        B: 输入张量，形状为 (K, N)
        
    Returns:
        输出张量，形状为 (M, N)
    """
    # 检查输入维度
    M, K_A = A.shape  # M=16384, K=32
    K_B, N = B.shape  # K=32, N=16384
    assert K_A == K_B, f"矩阵维度不匹配: A的列数{K_A} != B的行数{K_B}"
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 设置核心数
    num_cores = 20
    
    # 使用autotune自动选择最佳配置
    grid = lambda meta: (num_cores,)
    
    # 启动内核
    aikg_7_Matmul_with_small_K_dimension__kernel[grid](
        A, B, C,
        M, N, K_A,  # K_A作为编译时常量传入
        A.stride(0), A.stride(1),  # stride_am, stride_ak
        B.stride(0), B.stride(1),  # stride_bk, stride_bn
        C.stride(0), C.stride(1),  # stride_cm, stride_cn
        num_cores=num_cores,
        # BLOCK_M和BLOCK_N由autotune自动传入
    )
    
    return C