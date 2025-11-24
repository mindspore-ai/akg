import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 针对 16×16 小输入优化：数据量只有1KB，不需要核内切分
        # NUM_BLOCKS 核数不宜太多（避免核心空闲）
        triton.Config({'NUM_BLOCKS': 16}),  # 每核处理1行，Grid=16
        triton.Config({'NUM_BLOCKS': 8}),   # 每核处理2行，Grid=8
        triton.Config({'NUM_BLOCKS': 4}),   # 每核处理4行，Grid=4
        triton.Config({'NUM_BLOCKS': 2}),   # 每核处理8行，Grid=2
        triton.Config({'NUM_BLOCKS': 1}),   # 单核处理全部，Grid=1
    ],
    key=['M', 'N'],
)
@triton.jit
def div_kernel_row(
    dividend_ptr,
    divisor_ptr,
    output_ptr,
    M,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    """
    针对小输入优化的除法kernel（无核内切分）
    输入: dividend=(M, N), divisor=(1, N)
    输出: output=(M, N)
    
    策略：固定NUM_BLOCKS核数，每核直接处理分配的行，不需要核内循环
    """
    pid = tl.program_id(0)
    
    # 计算每个 block 需要处理的总行数
    rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS
    row_start = pid * rows_per_block
    row_end = tl.minimum(row_start + rows_per_block, M)
    
    # 计算当前核处理的行数
    num_rows = row_end - row_start
    
    # divisor 只需加载一次（所有行共享）
    offs_n = tl.arange(0, N)
    divisor = tl.load(divisor_ptr + offs_n)  # shape: (N,)
    
    # 直接处理所有分配的行（不需要循环）
    offs_m = row_start + tl.arange(0, 16)  # 假设最多16行
    mask_m = offs_m < row_end
    
    # 2D 索引
    offs_m_2d = offs_m[:, None]
    offs_n_2d = offs_n[None, :]
    
    # 计算全局偏移
    dividend_offs = offs_m_2d * N + offs_n_2d
    
    # 2D 掩码
    mask_2d = mask_m[:, None]
    
    # 加载 dividend
    dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)
    
    # 计算：divisor 自动广播到行维度
    output = dividend / divisor
    
    # 存储
    tl.store(output_ptr + dividend_offs, output, mask=mask_2d)


def custom_op_triton_torch(dividend: torch.Tensor, divisor: torch.Tensor) -> torch.Tensor:
    """
    Triton实现: 2D除法with广播 (针对16×16小输入优化)
    输入: dividend=(16, 16), divisor=(1, 16)
    输出: output=(16, 16)
    """
    assert dividend.ndim == 2 and divisor.ndim == 2, "Both inputs must be 2D"
    assert divisor.shape[0] == 1, "divisor must have shape (1, N)"
    assert dividend.shape[1] == divisor.shape[1], "Column dimension must match"
    
    dividend = dividend.contiguous()
    divisor = divisor.contiguous()
    output = torch.empty_like(dividend)
    
    M, N = dividend.shape
    
    # Grid 固定为 NUM_BLOCKS，由autotune选择最优核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    div_kernel_row[grid](
        dividend, divisor, output,
        M, N,
    )
    
    return output