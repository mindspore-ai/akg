import torch
import triton
import triton.language as tl


@triton.jit
def matmul_with_transposed_b_kernel(
    # 指针参数
    A_ptr: tl.tensor,
    B_ptr: tl.tensor,
    C_ptr: tl.tensor,
    # 张量维度
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    # 张量步幅
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    # 编译时常量
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr
):
    """
    Triton 内核实现矩阵乘法 A @ B.T
    
    参数:
        A_ptr: 指向A矩阵的指针
        B_ptr: 指向B矩阵的指针
        C_ptr: 指向输出矩阵的指针
        M, K, N: 矩阵维度 (编译时常量)
        stride_*: 各维度步幅 (编译时常量)
        BLOCK_DIM: 核心数 (编译时常量)
        TILE_*: 分块大小 (编译时常量)
    """
    # 获取当前程序ID (对应AUL核心ID)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate block start indices
    rstart = pid_m * TILE_M
    cstart = pid_n * TILE_N
    
    # 初始化累加器 (fp32)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
    # K维度流水线处理
    for k_block in range(0, K, TILE_K):
        # 创建块指针加载A分块
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(rstart, k_block),
            block_shape=(TILE_M, TILE_K),
            order=(1, 0)  # 行优先布局
        )
        a_tile = tl.load(a_block_ptr)
        
        # 创建块指针加载B分块
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            offsets=(cstart, k_block),
            block_shape=(TILE_N, TILE_K),
            order=(1, 0)  # 行优先布局
        )
        b_tile_raw = tl.load(b_block_ptr)
        
        # 转置B分块 (TILE_N, TILE_K) -> (TILE_K, TILE_N)
        b_tile_trans = tl.trans(b_tile_raw)
        
        # 执行矩阵乘法并累加 (fp16输入, fp32累加)
        acc = tl.dot(a_tile, b_tile_trans, acc)
    
    # 转换回fp16并存储结果
    c_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(rstart, cstart),
        block_shape=(TILE_M, TILE_N),
        order=(1, 0)  # 行优先布局
    )
    tl.store(c_block_ptr, acc)



# @triton.jit
# def matmul_with_transposed_b_kernel(
#     # 指针参数
#     A_ptr: tl.tensor,
#     B_ptr: tl.tensor,
#     C_ptr: tl.tensor,
#     # 张量维度
#     M: tl.constexpr,
#     K: tl.constexpr,
#     N: tl.constexpr,
#     # 张量步幅
#     stride_am: tl.constexpr,
#     stride_ak: tl.constexpr,
#     stride_bk: tl.constexpr,
#     stride_bn: tl.constexpr,
#     stride_cm: tl.constexpr,
#     stride_cn: tl.constexpr,
#     # 编译时常量
#     TILE_M: tl.constexpr,
#     TILE_N: tl.constexpr,
#     TILE_K: tl.constexpr
# ):
#     """
#     Triton 内核实现矩阵乘法 A @ B.T
    
#     参数:
#         A_ptr: 指向A矩阵的指针
#         B_ptr: 指向B矩阵的指针
#         C_ptr: 指向输出矩阵的指针
#         M, K, N: 矩阵维度 (编译时常量)
#         stride_*: 各维度步幅 (编译时常量)
#         BLOCK_DIM: 核心数 (编译时常量)
#         TILE_*: 分块大小 (编译时常量)
#     """
#     # 获取当前程序ID (对应AUL核心ID)
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
    
#     offs_am = pid_m * TILE_M + tl.arange(0, TILE_M)
#     # [0, 1, ..., TILE_M]
#     offs_bn = pid_n * TILE_N + tl.arange(0, TILE_N)
#     # [0, 1, ..., TILE_N]
#     offs_k = tl.arange(0, TILE_K)
#     # [0, 1, ..., TILE_K]
    
#     A_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     # offs_am[:, None]: [[0], [1], ..., [TILE_M]], shape = (TILE_M, 1)
#     # offs_k[None, :]: [[0, 1, ..., TILE_K]], shape = (1, TILE_K)
#     # broadcast add: shape = (TILE_M, TILE_K)
#     B_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
#     # offs_k[:, None]: [[0], [1], ..., [TILE_K]], shape = (TILE_K, 1)
#     # offs_n[None, :]: [[0, 1, ..., TILE_N]], shape = (1, TILE_N)
#     # broadcast add: shape = (TILE_K, TILE_N)
    
#     accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    
#     for k in range(0, tl.cdiv(K, TILE_K)):
#         a = tl.load(A_ptrs, mask=offs_k[None, :] < K - k * TILE_K, other=0.0)
#         b = tl.load(B_ptrs, mask=offs_k[:, None] < K - k * TILE_K, other=0.0)
#         accumulator += tl.dot(a, b)
#         A_ptrs += TILE_K * stride_ak
#         B_ptrs += TILE_K * stride_bk
    
#     C_ptrs = C_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
#     tl.store(C_ptrs, accumulator)


def matmul_with_transposed_b_triton_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton 矩阵乘法启动函数 (A @ B.T)
    
    参数:
        A: 输入张量, 形状 (M, K)
        B: 输入张量, 形状 (N, K)
        
    返回:
        C: 输出张量, 形状 (M, N)
    """
    # 验证输入形状
    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, f"K维度不匹配: {K} != {K2}"
    
    # 预分配输出张量
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # 硬编码参数 (来自AUL实现)
    TILE_M = 64
    TILE_N = 64
    TILE_K = 64
    
    # 计算张量步幅
    stride_am, stride_ak = A.stride(0), A.stride(1)
    stride_bn, stride_bk = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    # 配置网格 (一维网格, 大小为BLOCK_DIM)
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    grid = (grid_m, grid_n)
    
    # 启动内核
    matmul_with_transposed_b_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C,
        M=M, K=K, N=N,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K
    )
    
    return C
