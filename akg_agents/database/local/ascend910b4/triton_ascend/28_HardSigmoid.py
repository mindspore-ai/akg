import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384, 'VEC_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 8192, 'VEC_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 4096, 'VEC_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048, 'VEC_SIZE': 2048}),
    ],
    key=['N'],
)
@triton.jit
def aikg_28_HardSigmoid_kernel(
    x_ptr,
    y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    VEC_SIZE: tl.constexpr,
):
    """
    HardSigmoid激活函数内核实现：y = max(0, min(1, (x + 3) / 6))
    """
    # 获取程序ID
    pid = tl.program_id(0)
    
    # 计算当前块的数据范围
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # 加载数据到UB（Unified Buffer）
    x_tile = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 直接计算HardSigmoid，无需向量化循环
    # HardSigmoid计算：y = max(0, min(1, (x + 3) / 6))
    y_tile = (x_tile + 3.0) / 6.0
    y_tile = tl.maximum(y_tile, 0.0)
    y_tile = tl.minimum(y_tile, 1.0)
    
    # 存储结果到全局内存
    tl.store(y_ptr + offsets, y_tile, mask=mask)


def aikg_28_HardSigmoid_triton_ascend_torch(x: torch.Tensor) -> torch.Tensor:
    """
    HardSigmoid激活函数的Triton实现
    
    Args:
        x (torch.Tensor): 输入张量，任意形状
        
    Returns:
        torch.Tensor: 输出张量，与输入形状相同
    """
    # 确保输入张量是连续的
    if not x.is_contiguous():
        x = x.contiguous()
    
    # 获取输入张量的元素总数
    N = x.numel()
    
    # 创建输出张量
    y = torch.empty_like(x)
    
    # 定义grid大小（使用lambda函数以适应autotune）
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # 启动内核
    aikg_28_HardSigmoid_kernel[grid](
        x, y, N
        # BLOCK_SIZE和VEC_SIZE由autotune自动传入
    )
    
    return y