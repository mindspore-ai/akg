import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384}, num_cores=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_cores=32),
        triton.Config({'BLOCK_SIZE': 4096}, num_cores=64),
        triton.Config({'BLOCK_SIZE': 2048}, num_cores=128),
        triton.Config({'BLOCK_SIZE': 1024}, num_cores=256),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_32_HardTanh_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    HardTanh激活函数内核实现
    y = min(max(x, min_val), max_val)
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块的数据偏移
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < n_elements
    
    # 从全局内存加载数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # HardTanh计算: y = min(max(x, min_val), max_val)
    # 使用中间变量减少内存占用
    temp = tl.maximum(x, min_val)
    result = tl.minimum(temp, max_val)
    
    # 存储结果到全局内存
    tl.store(y_ptr + offsets, result, mask=mask)


def aikg_32_HardTanh_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    HardTanh激活函数Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，形状为(batch_size, dim)
        
    Returns:
        torch.Tensor: 输出张量，形状与输入相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入张量的形状和元素总数
    batch_size, dim = x.shape  # batch_size=16, dim=16384
    n_elements = x.numel()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 定义网格大小（使用lambda函数，autotune会自动调整）
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_32_HardTanh_kernel[grid](
        x, y, n_elements,
        min_val=-1.0, max_val=1.0
        # BLOCK_SIZE参数由autotune自动传入
    )
    
    return y