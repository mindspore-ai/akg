import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_27_SELU__kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    SELU激活函数Triton内核实现
    """
    # SELU参数
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块的数据范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据到UB
    x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 计算正部: max(0, x)
    pos_mask = x_tile > 0.0
    pos_part = tl.where(pos_mask, x_tile, 0.0)
    
    # 计算负部: alpha * (exp(x) - 1)
    neg_mask = x_tile <= 0.0
    temp_exp = tl.math.exp(x_tile)
    temp_exp = temp_exp - 1.0
    temp_exp = temp_exp * alpha
    neg_part = tl.where(neg_mask, temp_exp, 0.0)
    
    # 合并结果并缩放
    result_tile = pos_part + neg_part
    result_tile = result_tile * scale
    
    # 存储结果
    tl.store(y_ptr + offsets, result_tile, mask=mask)


def aikg_27_SELU__triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    SELU激活函数Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        
    Returns:
        torch.Tensor: 应用SELU后的输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 获取元素总数
    n_elements = x.numel()
    
    # 定义grid lambda函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_27_SELU__kernel[grid](
        x, output, n_elements
    )
    
    return output