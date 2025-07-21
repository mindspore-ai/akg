import torch
import triton
import triton.language as tl

@triton.jit
def matmul_with_transposed_a_kernel(
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
            shape=(K, M),
            strides=(stride_ak, stride_am),
            offsets=(k_block, rstart),
            block_shape=(TILE_K, TILE_M),
            order=(1, 0)  # 行优先布局
        )
        a_tile = tl.load(a_block_ptr)
        a_tile_trans = tl.trans(a_tile)
        
        # 创建块指针加载B分块
        b_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_block, cstart),
            block_shape=(TILE_K, TILE_N),
            order=(1, 0)  # 行优先布局
        )
        b_tile = tl.load(b_block_ptr)

        # 执行矩阵乘法并累加 (fp16输入, fp32累加)
        acc = tl.dot(a_tile_trans, b_tile, acc)

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


def matmul_with_transposed_a_triton_torch(A: torch.Tensor, B: torch.Tensor):
    """计算 A.T @ B 的 Triton 启动函数"""
    
    # 验证输入形状
    assert A.dim() == 2 and B.dim() == 2, "输入必须是2D张量"
    assert A.shape[0] == B.shape[0], "K 维度必须匹配"
    
    # 提取形状参数
    K, M = A.shape
    _, N = B.shape
    
    # 检查维度整除性
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    
    # 硬编码参数 (来自AUL实现)
    TILE_M = 64
    TILE_N = 64
    TILE_K = 64
    
    # 计算张量步幅
    stride_ak, stride_am = A.stride(0), A.stride(1)
    stride_bk, stride_bn = B.stride(0), B.stride(1)
    stride_cm, stride_cn = C.stride(0), C.stride(1)
    
    # 配置网格 (一维网格, 大小为BLOCK_DIM)
    grid_m = (M + TILE_M - 1) // TILE_M
    grid_n = (N + TILE_N - 1) // TILE_N
    grid = (grid_m, grid_n)
    
    # 启动内核
    matmul_with_transposed_a_kernel[grid](
        A_ptr=A, B_ptr=B, C_ptr=C,
        M=M, K=K, N=N,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
        TILE_M=TILE_M, TILE_N=TILE_N, TILE_K=TILE_K
    )
    
    return C