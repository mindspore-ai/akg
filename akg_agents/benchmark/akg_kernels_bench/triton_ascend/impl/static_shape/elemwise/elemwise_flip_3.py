import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数，SUB_M 控制内部每次处理行数
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 8}), #最优
        triton.Config({'NUM_BLOCKS': 512, 'SUB_M': 8}),
        triton.Config({'NUM_BLOCKS': 256, 'SUB_M': 8}),
    ],
    key=['M', 'N'],
)
@triton.jit
def flip_kernel(
    output_ptr,
    input_ptr,
    M: tl.constexpr,
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
    
    # 内层循环：每次处理 SUB_M 行
    for sub_start in range(row_start, row_end, SUB_M):
        # 当前子块的行索引
        offs_m = sub_start + tl.arange(0, SUB_M)
        mask_m = offs_m < row_end
        
        # 列偏移（固定处理所有 N 列）
        offs_n = tl.arange(0, N)
        
        # 2D 索引
        offs_m_2d = offs_m[:, None]
        offs_n_2d = offs_n[None, :]
        
        # 计算全局偏移
        input_offs = offs_m_2d * N + offs_n_2d
        # 2D 掩码
        mask_2d = mask_m[:, None]
        # 加载数据
        input_data = tl.load(input_ptr + input_offs, mask=mask_2d, other=0.0)
        output = tl.flip(input_data, 1)
        # 存储
        tl.store(output_ptr + input_offs, output, mask=mask_2d)


def custom_op_triton_torch(x: torch.Tensor):
    B, H, W = x.shape
    x_reshaped = x.reshape(-1, W)
    M, N = x_reshaped.shape
    output_reshaped = torch.empty_like(x_reshaped)
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    flip_kernel[grid](
        output_reshaped,
        x_reshaped,
        M, N,
    )
    output = output_reshaped.reshape(B, H, W)
    return output