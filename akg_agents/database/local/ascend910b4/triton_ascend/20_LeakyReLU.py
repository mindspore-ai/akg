import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32768, 'TILE_SIZE': 16384}),
        triton.Config({'BLOCK_SIZE': 16384, 'TILE_SIZE': 8192}),
        triton.Config({'BLOCK_SIZE': 8192, 'TILE_SIZE': 4096}),
        triton.Config({'BLOCK_SIZE': 4096, 'TILE_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 2048, 'TILE_SIZE': 1024}),
    ],
    key=['N'],
)
@triton.jit
def aikg_20_LeakyReLU_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    negative_slope: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    """
    LeakyReLU Triton 内核实现
    """
    pid = tl.program_id(0)
    
    # 计算当前程序块负责的数据范围
    block_start = pid * BLOCK_SIZE
    block_end = min(block_start + BLOCK_SIZE, n_elements)
    
    # 分块处理当前程序块的数据
    for tile_start in range(block_start, block_end, TILE_SIZE):
        tile_end = min(tile_start + TILE_SIZE, block_end)
        tile_size = tile_end - tile_start
        
        # 计算偏移和掩码
        offsets = tile_start + tl.arange(0, TILE_SIZE)
        mask = offsets < n_elements
        
        # 加载输入数据
        x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # 计算 LeakyReLU: max(x, negative_slope * x)
        scaled_tile = x_tile * negative_slope
        y_tile = tl.maximum(x_tile, scaled_tile)
        
        # 存储结果
        tl.store(y_ptr + offsets, y_tile, mask=mask)


def aikg_20_LeakyReLU_triton_ascend_torch(x: torch.Tensor, negative_slope: float = 0.01):
    """
    LeakyReLU Triton 启动函数
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        negative_slope (float, optional): LeakyReLU负斜率参数，默认为0.01
        
    Returns:
        torch.Tensor: 应用LeakyReLU后的输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 获取元素总数
    n_elements = x.numel()
    
    # 定义网格大小计算函数
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_20_LeakyReLU_kernel[grid](
        x, y, n_elements,
        negative_slope=negative_slope,
    )
    
    return y