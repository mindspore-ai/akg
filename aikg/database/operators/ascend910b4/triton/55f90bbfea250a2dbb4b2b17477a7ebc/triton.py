import torch
import triton
import triton.language as tl

@triton.jit
def matrix_scalar_multiplication_kernel(
    output_ptr,
    input_ptr,
    s_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton内核：矩阵标量乘法
    
    参数:
    output_ptr: 输出矩阵指针
    input_ptr: 输入矩阵指针
    s_ptr: 标量值指针
    total_elements: 矩阵总元素数
    BLOCK_SIZE: 每次处理的块大小
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    # 计算当前核心处理的数据范围
    elements_per_core = (total_elements + num_programs - 1) // num_programs
    start_idx = pid * elements_per_core
    end_idx = tl.minimum(start_idx + elements_per_core, total_elements)
    num_elements_this_core = end_idx - start_idx
    
    # 加载标量值
    s_val = tl.load(s_ptr)
    
    # 分块处理数据
    for i in range(0, num_elements_this_core, BLOCK_SIZE):
        offsets = start_idx + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end_idx
        
        # 加载输入数据
        data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        
        # 执行标量乘法
        result = data * s_val
        
        # 存储结果
        tl.store(output_ptr + offsets, result, mask=mask)


def matrix_scalar_multiplication_triton_torch(A: torch.Tensor, s: float) -> torch.Tensor:
    """
    Triton启动函数：矩阵标量乘法
    
    参数:
    A: 输入矩阵 (M, N)
    s: 标量值
    
    返回:
    C: 结果矩阵 (M, N)
    """
    # 确保输入张量是连续的
    A_contiguous = A.contiguous()
    
    # 创建标量张量
    s_tensor = torch.tensor([s], dtype=A.dtype, device=A.device)
    
    # 准备输出张量
    C = torch.empty_like(A_contiguous)
    
    # 展平矩阵为一维
    input_flat = A_contiguous.view(-1)
    output_flat = C.view(-1)
    total_elements = input_flat.numel()
    
    # 设置网格大小（核心数）
    BLOCK_DIM = 16
    grid = (BLOCK_DIM,)
    
    # 设置块大小（每次处理元素数）
    BLOCK_SIZE = 1024
    
    # 启动内核
    matrix_scalar_multiplication_kernel[grid](
        output_flat,
        input_flat,
        s_tensor,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return C
