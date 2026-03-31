import torch
import triton
import triton.language as tl

@triton.jit
def aikg_15_Matmul_for_lower_triangular_matrices_kernel(
    A_ptr,  # 输入矩阵A指针
    B_ptr,  # 输入矩阵B指针
    C_ptr,  # 输出矩阵C指针
    N,      # 矩阵维度
    BLOCK_SIZE: tl.constexpr,  # 块大小
):
    """
    下三角矩阵乘法Triton内核
    每个程序处理BLOCK_SIZE个元素
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块处理的起始和结束索引
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, N * N)
    
    # 处理当前块的所有元素
    for linear_idx in range(start_idx, end_idx):
        # 将1D索引转换为2D坐标
        i = linear_idx // N
        j = linear_idx % N
        
        # 只计算下三角部分
        if j <= i:
            # 初始化累加器为标量值
            acc = 0.0
            
            # 矩阵乘法内积计算
            for k in range(0, min(i, j) + 1):  # 只访问下三角元素
                # 加载A[i, k]和B[k, j]
                a_offset = i * N + k
                b_offset = k * N + j
                
                # 边界检查
                if k < N:
                    a_val = tl.load(A_ptr + a_offset)
                    b_val = tl.load(B_ptr + b_offset)
                    
                    # 乘积累加
                    acc += a_val * b_val
            
            # 存储结果
            tl.store(C_ptr + linear_idx, acc)
        else:
            # 上三角部分设为0
            tl.store(C_ptr + linear_idx, 0.0)


def aikg_15_Matmul_for_lower_triangular_matrices_triton_ascend_torch(A, B):
    """
    Triton下三角矩阵乘法启动函数
    
    Args:
        A (torch.Tensor): 下三角矩阵，形状为 (N, N)
        B (torch.Tensor): 下三角矩阵，形状为 (N, N)
        
    Returns:
        torch.Tensor: 下三角矩阵乘法结果，形状为 (N, N)
    """
    # 检查输入形状
    assert A.shape == B.shape, "输入矩阵形状必须相同"
    N = A.shape[0]  # 矩阵维度
    
    # 分配输出张量
    C = torch.empty((N, N), dtype=A.dtype, device=A.device)
    
    # 设置块大小
    BLOCK_SIZE = 256
    
    # 计算网格大小
    total_elements = N * N
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # 检查grid大小是否超过限制
    if grid_size > 65535:
        # 如果grid太大，使用固定核心数启动
        num_cores = 20  # Ascend 910B4有20个AI Core
        grid_size = min(grid_size, num_cores)
    
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    
    # 启动内核
    aikg_15_Matmul_for_lower_triangular_matrices_kernel[(grid_size,)](
        A, B, C, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return C