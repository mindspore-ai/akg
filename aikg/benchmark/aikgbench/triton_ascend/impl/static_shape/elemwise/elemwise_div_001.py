import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数（32 的倍数），SUB_M 控制内部每次处理行数
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 4}),
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 8}),
        triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 2}),
        triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 4}),
        triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 8}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 4}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 8}),
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
    SUB_M: tl.constexpr,
):
    # 固定 NUM_BLOCKS 个 blocks，每个 block 处理多行
    # 内部用 SUB_M 切分，减少寄存器压力
    pid = tl.program_id(0)
    
    # 计算每个 block 需要处理的总行数（手动 cdiv）
    rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS
    row_start = pid * rows_per_block
    row_end = tl.minimum(row_start + rows_per_block, M)
    
    # divisor 只需加载一次（所有行共享）
    offs_n = tl.arange(0, N)
    divisor = tl.load(divisor_ptr + offs_n)  # shape: (N,)
    
    # 内层循环：每次处理 SUB_M 行
    for sub_start in range(row_start, row_end, SUB_M):
        # 当前子块的行索引
        offs_m = sub_start + tl.arange(0, SUB_M)
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
    assert dividend.ndim == 2 and divisor.ndim == 2, "Both inputs must be 2D"
    assert divisor.shape[0] == 1, "divisor must have shape (1, N)"
    assert dividend.shape[1] == divisor.shape[1], "Column dimension must match"
    
    dividend = dividend.contiguous()
    divisor = divisor.contiguous()
    output = torch.empty_like(dividend)
    
    M, N = dividend.shape
    
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    div_kernel_row[grid](
        dividend, divisor, output,
        M, N,
    )
    
    return output