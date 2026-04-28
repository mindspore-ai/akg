import torch
import triton
import triton.language as tl

@triton.jit
def aikg_6_Matmul_with_large_K_dimension__kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,  # A的stride: [stride_am, stride_ak]
    stride_bk, stride_bn,  # B的stride: [stride_bk, stride_bn]
    stride_cm, stride_cn,  # C的stride: [stride_cm, stride_cn]
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_cores: tl.constexpr,
):
    """
    Triton矩阵乘法内核，针对大K维度优化
    """
    # 获取核心ID
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
        
        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        # K维度循环
        for k_outer in range(0, K, BLOCK_K):
            # 计算当前K块的起始位置
            start_k = k_outer
            
            # 创建A块的块指针
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(M, K),
                strides=(stride_am, stride_ak),
                offsets=(start_m, start_k),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0)
            )
            
            # 创建B块的块指针
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(start_k, start_n),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0)
            )
            
            # 加载A块
            a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
            
            # 加载B块
            b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
            
            # 矩阵乘法累加
            accumulator += tl.dot(a_tile, b_tile)
        
        # 创建C块的块指针
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(start_m, start_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0)
        )
        
        # 存储结果
        tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))


def aikg_6_Matmul_with_large_K_dimension__triton_ascend_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton矩阵乘法启动函数
    
    Args:
        A: 输入张量，形状为 (M, K)
        B: 输入张量，形状为 (K, N)
        
    Returns:
        输出张量，形状为 (M, N)
    """
    # 检查输入维度
    M, K_A = A.shape
    K_B, N = B.shape
    assert K_A == K_B, f"矩阵维度不匹配: A的K维度{K_A} != B的K维度{K_B}"
    K = K_A
    
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 分配输出张量
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    
    # 获取stride信息
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # 使用lambda函数定义grid
    num_cores = 20
    grid = (num_cores,)
    
    aikg_6_Matmul_with_large_K_dimension__kernel[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=512, num_cores=num_cores,
    )
    
    return C