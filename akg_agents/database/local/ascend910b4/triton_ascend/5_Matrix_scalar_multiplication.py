import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
    ],
    key=['total_elements'],
)
@triton.jit
def aikg_5_Matrix_scalar_multiplication_kernel(
    A_ptr,
    C_ptr,
    scalar_value,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton 矩阵标量乘法内核
    计算 C = A * scalar_value
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块处理的起始偏移
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < total_elements
    
    # 从全局内存加载数据
    A_data = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    
    # 执行标量乘法计算
    C_data = A_data * scalar_value
    
    # 将结果存储回全局内存
    tl.store(C_ptr + offsets, C_data, mask=mask)


def aikg_5_Matrix_scalar_multiplication_triton_ascend_torch(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Triton 矩阵标量乘法启动函数
    
    Args:
        A: 输入矩阵，形状为 (M, N)
        s: 标量值
        
    Returns:
        C: 结果矩阵，形状为 (M, N)
    """
    # 确保输入张量是连续的
    if not A.is_contiguous():
        A = A.contiguous()
    
    # 获取输入矩阵的形状
    M, N = A.shape  # M=16384, N=4096
    total_elements = M * N
    
    # 创建输出张量
    C = torch.empty_like(A)
    
    # 定义网格大小计算函数
    def grid(meta):
        return (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_5_Matrix_scalar_multiplication_kernel[grid](
        A, C, s, total_elements
    )
    
    return C