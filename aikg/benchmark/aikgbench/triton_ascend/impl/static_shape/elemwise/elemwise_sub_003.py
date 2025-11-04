import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数，SUB_M 控制内部每次处理行数，BLOCK_N 控制列切分大小
        # 对 N=131072 进行列切分，避免 UB 溢出
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 2, 'BLOCK_N': 4096}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 1, 'BLOCK_N': 16384}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 2, 'BLOCK_N': 8192}),
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 2, 'BLOCK_N': 8192}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 4, 'BLOCK_N': 4096}),
    ],
    key=['M', 'N'],
)
@triton.jit
def sub_kernel_row(
    input1_ptr,
    input2_ptr,
    output_ptr,
    M,
    N,
    NUM_BLOCKS: tl.constexpr,
    SUB_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 固定 NUM_BLOCKS 个 blocks，每个 block 处理多行
    # 内部用 SUB_M 切分行，用 BLOCK_N 切分列，减少寄存器压力和避免 UB 溢出
    pid = tl.program_id(0)
    
    # 计算每个 block 需要处理的总行数（手动 cdiv）
    rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS
    row_start = pid * rows_per_block
    row_end = tl.minimum(row_start + rows_per_block, M)
    
    # 外层循环：每次处理 SUB_M 行
    for sub_start in range(row_start, row_end, SUB_M):
        # 当前子块的行索引
        offs_m = sub_start + tl.arange(0, SUB_M)
        mask_m = offs_m < row_end
        
        # 内层循环：每次处理 BLOCK_N 列，避免 UB 溢出
        for col_start in range(0, N, BLOCK_N):
            col_end = tl.minimum(col_start + BLOCK_N, N)
            
            # 列偏移
            offs_n = col_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < col_end
            
            # 2D 索引
            offs_m_2d = offs_m[:, None]
            offs_n_2d = offs_n[None, :]
            
            # 计算全局偏移
            input1_offs = offs_m_2d * N + offs_n_2d
            input2_offs = offs_m_2d  # input2 是 (M, 1)，只有行索引
            
            # 2D 掩码
            mask_2d = mask_m[:, None] & mask_n[None, :]
            
            # 加载数据
            input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)
            input2 = tl.load(input2_ptr + input2_offs, mask=mask_m[:, None], other=0.0)
            
            # 计算：input2 自动广播到列维度
            output = input1 - input2
            
            # 存储
            tl.store(output_ptr + input1_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.ndim == 2 and input2.ndim == 2, "Both inputs must be 2D"
    assert input1.shape[0] == input2.shape[0], "First dimension must match"
    assert input2.shape[1] == 1, "input2 must have shape (M, 1)"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)
    
    M, N = input1.shape
    
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    sub_kernel_row[grid](
        input1, input2, output,
        M, N,
    )
    
    return output