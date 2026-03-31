import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'NUM_CORES': 20, 'BLOCK_SIZE': 8192}),
        triton.Config({'NUM_CORES': 10, 'BLOCK_SIZE': 16384}),
        triton.Config({'NUM_CORES': 40, 'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def aikg_25_Swish_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    NUM_CORES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Swish激活函数内核：y = x * sigmoid(x)
    使用固定核心数启动，每个核心循环处理多个数据块
    """
    pid = tl.program_id(0)  # 核心ID: 0~NUM_CORES-1
    
    # 计算每个核心需要处理的元素总数
    ELEMENTS_PER_CORE = tl.cdiv(n_elements, NUM_CORES)
    # 向上取整到BLOCK_SIZE的倍数
    ELEMENTS_PER_CORE = tl.cdiv(ELEMENTS_PER_CORE, BLOCK_SIZE) * BLOCK_SIZE
    
    # 计算当前核心负责的数据范围
    core_start = pid * ELEMENTS_PER_CORE
    core_end = min(core_start + ELEMENTS_PER_CORE, n_elements)
    
    # 分块处理当前核心负责的数据
    for block_start in range(core_start, core_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < core_end
        
        # 加载输入数据
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # 计算swish: x * sigmoid(x) = x / (1 + exp(-x))
        neg_x = -x
        exp_neg_x = tl.exp(neg_x)
        one_plus_exp = 1.0 + exp_neg_x
        result = x / one_plus_exp
        
        # 存储结果
        tl.store(y_ptr + offsets, result, mask=mask)


def aikg_25_Swish_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Swish激活函数Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，形状为(batch_size, dim)
        
    Returns:
        torch.Tensor: 输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入张量的元素总数
    n_elements = x.numel()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 启动内核 - 使用lambda函数设置grid
    grid = lambda meta: (meta['NUM_CORES'],)
    aikg_25_Swish_kernel[grid](
        x, y, n_elements
        # NUM_CORES和BLOCK_SIZE由autotune自动传入
    )
    
    return y


# 测试代码
if __name__ == "__main__":
    # 固定随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 获取输入数据
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    
    # 运行Triton实现
    output = aikg_25_Swish_triton_ascend_torch(x)
    
    # 验证结果
    expected = x * torch.sigmoid(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"最大误差: {torch.max(torch.abs(output - expected))}")
    print(f"结果验证: {torch.allclose(output, expected, atol=1e-6)}")