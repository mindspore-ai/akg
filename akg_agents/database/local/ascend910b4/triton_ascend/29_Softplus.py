import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192, 'VEC_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 16384, 'VEC_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_SIZE': 64}),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_29_Softplus_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    Softplus激活函数内核实现：y = log(1 + exp(x))
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块处理的偏移范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < n_elements
    
    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 计算Softplus：y = log(1 + exp(x))
    exp_x = tl.math.exp(x)
    one_plus_exp = 1.0 + exp_x
    y = tl.math.log(one_plus_exp)
    
    # 存储结果
    tl.store(y_ptr + offsets, y, mask=mask)


def aikg_29_Softplus_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Softplus激活函数的Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，形状为(batch_size, dim)
        
    Returns:
        torch.Tensor: 输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入张量的总元素数
    n_elements = x.numel()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 定义grid大小（使用lambda函数以适应autotune）
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_29_Softplus_kernel[grid](
        x, y, n_elements,
        # BLOCK_SIZE和VEC_SIZE由autotune自动传入
    )
    
    return y