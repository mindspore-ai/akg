import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192, 'NUM_CORES': 40}),
        triton.Config({'BLOCK_SIZE': 4096, 'NUM_CORES': 40}),
        triton.Config({'BLOCK_SIZE': 2048, 'NUM_CORES': 40}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_CORES': 40}),
    ],
    key=['total_elements'],
)
@triton.jit
def aikg_88_MinGPTNewGelu_kernel(
    x_ptr,
    y_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """
    Triton GELU 激活函数内核
    实现公式: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    # 预计算常量
    sqrt_2_pi = 0.7978845608028654
    alpha = 0.044715
    half = 0.5
    one = 1.0
    
    # 获取核心ID
    core_idx = tl.program_id(0)
    
    # 计算每个核心处理的元素总数
    elements_per_core = tl.cdiv(total_elements, NUM_CORES)
    start_idx = core_idx * elements_per_core
    end_idx = tl.minimum(start_idx + elements_per_core, total_elements)
    
    # 循环处理当前核心负责的数据块
    for block_start in range(start_idx, end_idx, BLOCK_SIZE):
        block_end = tl.minimum(block_start + BLOCK_SIZE, end_idx)
        current_block_size = block_end - block_start
        
        # 计算当前块的偏移
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_end
        
        # 加载输入数据
        x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # GELU计算流程
        # 计算 x^3 = x * x * x
        x_squared = x_tile * x_tile
        x_cubed = x_squared * x_tile
        
        # 计算 0.044715 * x^3 + x
        temp = alpha * x_cubed + x_tile
        
        # 计算 sqrt(2/pi) * (x + 0.044715 * x^3)
        temp = sqrt_2_pi * temp
        
        # 计算 tanh(...) + 1.0
        temp = tl.math.tanh(temp) + one
        
        # 计算最终结果: 0.5 * x * (1.0 + tanh(...))
        result = half * x_tile * temp
        
        # 存储结果
        tl.store(y_ptr + offsets, result, mask=mask)


def aikg_88_MinGPTNewGelu_triton_ascend_torch(x):
    """
    Triton GELU 激活函数启动函数
    
    参数:
        x: 输入张量，形状为 [batch_size, dim] 或 [total_elements]
        
    返回:
        y: 输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 展平输入张量为一维
    x_flat = x.view(-1)
    total_elements = x_flat.numel()
    
    # 分配输出张量
    y = torch.empty_like(x_flat)
    
    # 设置核心数（从autotune配置中获取，这里设置为40）
    NUM_CORES = 40
    
    # 启动内核
    aikg_88_MinGPTNewGelu_kernel[(NUM_CORES,)](
        x_flat, y, total_elements,
        # BLOCK_SIZE 和 NUM_CORES 由 autotune 自动传入
    )
    
    # 恢复原始形状
    return y.view_as(x)