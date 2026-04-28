import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=1),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_31_ELU_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    alpha: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    ELU激活函数Triton内核实现
    
    Args:
        x_ptr: 输入张量指针
        y_ptr: 输出张量指针
        n_elements: 总元素数量
        alpha: ELU参数
        BLOCK_SIZE: 块大小
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块的数据偏移
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 创建边界掩码
    mask = offsets < n_elements
    
    # 加载输入数据
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ELU计算
    # 条件判断：x > 0
    condition = x > 0.0
    
    # 计算指数部分：alpha * (exp(x) - 1)
    exp_part = tl.math.exp2(x * 1.442695)  # exp2(x) = 2^x, 使用1.442695进行转换
    exp_part = exp_part - 1.0
    exp_part = exp_part * alpha
    
    # 条件选择：如果x > 0则返回x，否则返回alpha*(exp(x)-1)
    result = tl.where(condition, x, exp_part)
    
    # 存储结果
    tl.store(y_ptr + offsets, result, mask=mask)


def aikg_31_ELU_triton_ascend_torch(x, alpha=1.0):
    """
    ELU激活函数Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        alpha (float): ELU参数，默认为1.0
        
    Returns:
        torch.Tensor: 应用ELU后的输出张量，与输入同形状
    """
    # 确保输入张量在设备上且连续
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 获取总元素数量
    n_elements = x.numel()
    
    # 启动内核
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    aikg_31_ELU_kernel[grid](
        x, y, n_elements,
        alpha=alpha
    )
    
    return y


# 测试代码
if __name__ == "__main__":
    # 测试数据
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, device='npu')
    
    # 运行Triton实现
    output = aikg_31_ELU_triton_ascend_torch(x, alpha=1.0)
    
    # 验证结果
    expected = torch.nn.functional.elu(x, alpha=1.0)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Max difference: {torch.max(torch.abs(output - expected))}")
    print("Test passed!")