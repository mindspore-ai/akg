import torch
import triton
import triton.language as tl

@triton.jit
def tall_skinny_matrix_multiplication__kernel(
    # 张量指针参数
    C_ptr, A_ptr, B_ptr,
    # 矩阵维度和步丐参数
    M: tl.constexpr, N: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    # 编译时常量参数
    BLOCK_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    num_outer_loops: tl.constexpr, num_inner_loops: tl.constexpr
):
    """
    Triton内核实现高瘦矩阵乘法
    
    参数说明:
    C_ptr - 输出矩阵指针
    A_ptr - 输入矩阵A指针 (形状 M x N)
    B_ptr - 输入矩阵B指针 (形状 N x M)
    M, N - 矩阵维度
    stride_* - 矩阵各维度步丐
    BLOCK_* - 块大小参数
    num_outer_loops - 外层循环次数
    num_inner_loops - 内层循环次数
    """
    
    # 获取当前程序ID (核索引)
    pid = tl.program_id(0)
    
    # 计算每个核处理的行数
    rows_per_core = M // BLOCK_DIM
    
    # 计算当前核负责的起始行
    start_row = pid * rows_per_core
    
    # 外层循环处理行块
    for i_outer in tl.static_range(num_outer_loops):
        # 计算当前行块起始索引
        current_i = start_row + i_outer * BLOCK_M
        
        # 创建A矩阵块指针 (BLOCK_M x N)
        a_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, N),
            strides=(stride_am, stride_ak),
            offsets=(current_i, 0),
            block_shape=(BLOCK_M, N),
            order=(1, 0)  # 行优先
        )
        
        # 加载A矩阵块
        a_block = tl.load(a_block_ptr, boundary_check=(0, 1))
        
        # 内层循环处理列块
        for j_inner in tl.static_range(num_inner_loops):
            # 计算当前列块起始索引
            current_j = j_inner * BLOCK_N
            
            # 创建B矩阵块指针 (N x BLOCK_N)
            b_block_ptr = tl.make_block_ptr(
                base=B_ptr,
                shape=(N, M),
                strides=(stride_bk, stride_bn),
                offsets=(0, current_j),
                block_shape=(N, BLOCK_N),
                order=(1, 0)  # 行优先
            )
            
            # 加载B矩阵块
            b_block = tl.load(b_block_ptr, boundary_check=(0, 1))
            
            # 执行矩阵乘法 (BLOCK_M x N) @ (N x BLOCK_N) = (BLOCK_M x BLOCK_N)
            c_block = tl.dot(a_block, b_block)
            
            # 创建C矩阵块指针
            c_block_ptr = tl.make_block_ptr(
                base=C_ptr,
                shape=(M, M),
                strides=(stride_cm, stride_cn),
                offsets=(current_i, current_j),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0)  # 行优先
            )
            
            # 存储计算结果
            tl.store(c_block_ptr, c_block, boundary_check=(0, 1))


def tall_skinny_matrix_multiplication__triton_torch(A, B):
    """
    Triton启动函数 - 高瘦矩阵乘法
    
    参数:
    A (torch.Tensor): 输入矩阵 (M x N)
    B (torch.Tensor): 输入矩阵 (N x M)
    
    返回:
    torch.Tensor: 输出矩阵 (M x M)
    """
    # 确保内存布局为行优先
    A = A.contiguous()
    B = B.contiguous()
    
    # 获取矩阵维度
    M, N = A.shape
    
    # 验证矩阵形状
    assert B.shape == (N, M), f"B矩阵形状应为({N}, {M}), 实际为{B.shape}"
    
    # 创建输出张量
    C = torch.empty((M, M), device=A.device, dtype=A.dtype)
    
    # 获取矩阵步丐 (以元素为单位)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = B.stride()
    stride_cm, stride_cn = C.stride()
    
    # 硬编码参数 (与AUL代码一致)
    BLOCK_DIM = 16  # 核数量
    BLOCK_M = 64    # 行块大小
    BLOCK_N = 256   # 列块大小
    
    # 计算循环次数
    rows_per_core = M // BLOCK_DIM
    num_outer_loops = rows_per_core // BLOCK_M
    num_inner_loops = M // BLOCK_N  # 基于M维度切分
    
    # 验证参数整除关系
    assert M % BLOCK_DIM == 0, "M必须能被BLOCK_DIM整除"
    assert rows_per_core % BLOCK_M == 0, "rows_per_core必须能被BLOCK_M整除"
    assert M % BLOCK_N == 0, "M必须能被BLOCK_N整除"
    
    # 配置网格 (一维网格，大小=BLOCK_DIM)
    grid = (BLOCK_DIM,)
    
    # 启动内核
    tall_skinny_matrix_multiplication__kernel[grid](
        # 张量指针
        C, A, B,
        # 矩阵维度和步丐
        M, N,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # 编译时常量
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_outer_loops=num_outer_loops,
        num_inner_loops=num_inner_loops
    )
    
    return C