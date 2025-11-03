import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 16}), # 最优，核数被shape整除
        # 40 核数（匹配硬件）
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 16}), # 性能接近最优
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 32}), # 超ub
    ],
    key=['M', 'N'],
)
@triton.jit
def maximum_kernel_row(
    input1_ptr,
    input2_ptr,
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
    
    # input2 只需加载一次（所有行共享，第一维广播）
    offs_n = tl.arange(0, N)
    input2 = tl.load(input2_ptr + offs_n)  # shape: (N,)
    
    # 内层循环：每次处理 SUB_M 行
    for sub_start in range(row_start, row_end, SUB_M):
        # 当前子块的行索引
        offs_m = sub_start + tl.arange(0, SUB_M)
        mask_m = offs_m < row_end
        
        # 2D 索引
        offs_m_2d = offs_m[:, None]
        offs_n_2d = offs_n[None, :]
        
        # 计算全局偏移
        input1_offs = offs_m_2d * N + offs_n_2d
        
        # 2D 掩码
        mask_2d = mask_m[:, None]
        
        # 加载 input1
        input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)
        
        # 计算：input2 自动广播到行维度
        output = tl.maximum(input1, input2)
        
        # 存储
        tl.store(output_ptr + input1_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.ndim == 2 and input2.ndim == 2, "Both inputs must be 2D"
    assert input2.shape[0] == 1, "input2 must have shape (1, N) for row broadcasting"
    assert input1.shape[1] == input2.shape[1], "Column dimension must match"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)
    
    M, N = input1.shape
    
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    maximum_kernel_row[grid](
        input1, input2, output,
        M, N,
    )
    
    return output