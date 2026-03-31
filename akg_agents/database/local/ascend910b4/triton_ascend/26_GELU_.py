import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'VEC_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512, 'VEC_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 256, 'VEC_SIZE': 64}),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_26_GELU__kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    GELU激活函数Triton内核实现
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块处理的偏移范围
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据到UB（Unified Buffer）
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU计算：0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    # 数值稳定实现
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    gelu_const = 0.044715
    
    # 计算x^3
    x_cubed = x * x * x
    
    # 计算内部表达式
    inner = sqrt_2_over_pi * (x + gelu_const * x_cubed)
    
    # 使用tanh近似（Triton内置的数学函数）
    tanh_inner = tl.tanh(inner)
    
    # 最终GELU计算
    result = 0.5 * x * (1.0 + tanh_inner)
    
    # 存储结果
    tl.store(y_ptr + offsets, result, mask=mask)


def aikg_26_GELU__triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    GELU激活函数Triton实现 - 启动函数
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        
    Returns:
        torch.Tensor: 应用GELU后的输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 获取总元素数
    n_elements = x.numel()
    
    # 使用lambda函数定义grid，支持autotune
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_26_GELU__kernel[grid](
        x, output, n_elements
        # BLOCK_SIZE和VEC_SIZE由autotune自动传入
    )
    
    return output


# 测试代码
if __name__ == "__main__":
    # 创建测试输入
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim, device='npu')
    
    # 运行Triton实现
    output_triton = aikg_26_GELU__triton_ascend_torch(x)
    
    # 运行PyTorch原生实现进行验证
    output_torch = torch.nn.functional.gelu(x)
    
    # 检查结果一致性
    print(f"Max difference: {torch.max(torch.abs(output_triton - output_torch))}")
    print(f"Results match: {torch.allclose(output_triton, output_torch, atol=1e-6)}")