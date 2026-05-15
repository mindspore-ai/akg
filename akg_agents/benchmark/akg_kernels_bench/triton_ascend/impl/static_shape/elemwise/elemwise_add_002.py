import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 8, 'SUB_N': 1024}),
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 4, 'SUB_N': 2048}),
        triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 2, 'SUB_N': 2048}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 4, 'SUB_N': 1024}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 2, 'SUB_N': 2048}),
        triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 4, 'SUB_N': 2048}),
    ],
    key=['M', 'N'],
)
@triton.jit
def add_kernel_broadcast_row(
    input1_ptr,
    input2_ptr,
    output_ptr,
    M,
    N,
    NUM_BLOCKS: tl.constexpr,
    SUB_M: tl.constexpr,
    SUB_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 每个 block 负责的行范围
    rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS
    row_start = pid * rows_per_block
    row_end = tl.minimum(row_start + rows_per_block, M)
    
    # 外层：切分行（每次 SUB_M 行）
    for sub_m_start in range(row_start, row_end, SUB_M):
        offs_m = sub_m_start + tl.arange(0, SUB_M)
        mask_m = offs_m < row_end  # shape: (SUB_M,)

        # 内层：切分列（每次 SUB_N 列）
        for sub_n_start in range(0, N, SUB_N):
            offs_n = sub_n_start + tl.arange(0, SUB_N)
            mask_n = offs_n < N  # shape: (SUB_N,)

            # 2D 索引
            offs_m_2d = offs_m[:, None]   # (SUB_M, 1)
            offs_n_2d = offs_n[None, :]   # (1, SUB_N)

            # input1 偏移: (M, N)
            input1_offs = offs_m_2d * N + offs_n_2d
            mask_2d = mask_m[:, None] & mask_n[None, :]

            # 加载 input1
            input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)

            # 加载 input2: (1, N) → 只需列索引（与行无关）
            input2 = tl.load(input2_ptr + offs_n, mask=mask_n, other=0.0)  # (SUB_N,)

            # 广播加法: (SUB_M, SUB_N) + (SUB_N,) → 自动广播
            output = input1 + input2[None, :]  # shape: (SUB_M, SUB_N)

            # 存储
            tl.store(output_ptr + input1_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.ndim == 2 and input2.ndim == 2, "Both inputs must be 2D"
    assert input2.shape[0] == 1, "input2 must have shape (1, N)"
    assert input1.shape[1] == input2.shape[1], "N must match"

    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)

    M, N = input1.shape  # M=131072, N=2048

    grid = lambda meta: (meta['NUM_BLOCKS'],)

    add_kernel_broadcast_row[grid](
        input1, input2, output,
        M, N,
    )
    return output