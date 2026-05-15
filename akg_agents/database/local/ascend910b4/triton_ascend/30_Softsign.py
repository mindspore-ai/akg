import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_30_Softsign_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Softsign激活函数内核：y = x / (1 + |x|)
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块的数据偏移
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < n_elements
    
    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 计算绝对值 |x|
    abs_x = tl.abs(x)
    
    # 计算分母 1 + |x|
    denominator = 1.0 + abs_x
    
    # 计算最终结果 x / (1 + |x|)
    result = x / denominator
    
    # 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)


def aikg_30_Softsign_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Softsign激活函数的Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        
    Returns:
        torch.Tensor: 输出张量，与输入相同形状，应用了Softsign激活
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入元素总数
    n_elements = x.numel()
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 定义grid大小计算函数
    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_30_Softsign_kernel[grid](
        x, output, n_elements
    )
    
    return output