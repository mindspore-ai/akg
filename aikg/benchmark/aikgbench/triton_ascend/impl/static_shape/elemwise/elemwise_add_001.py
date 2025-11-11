import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # 针对 16×2048 输入优化：行少列多，切分列维度 + 核内切分
        # Grid = (M, cdiv(N, BLOCK_N))
        triton.Config({'BLOCK_N': 2048, 'SUB_N': 2048}),   # Grid=(16, 1) = 16核，每核循环4次
        triton.Config({'BLOCK_N': 2048, 'SUB_N': 1024}),   # Grid=(16, 1) = 16核，每核循环8次
        triton.Config({'BLOCK_N': 1024, 'SUB_N': 1024}),  # Grid=(16, 2) = 32核，每核循环4次
    ],
    key=['M', 'N'],
)
@triton.jit
def add_broadcast_kernel(
    input1_ptr,
    input2_ptr,
    output_ptr,
    M: tl.constexpr,
    N,
    stride_in1_m,
    stride_in2_n,
    stride_out_m,
    BLOCK_N: tl.constexpr,
    SUB_N: tl.constexpr,
):
    """
    广播加法kernel：(M, N) + (1, N) -> (M, N)
    输入1: (16, 2048)
    输入2: (1, 2048) - 广播第0维
    输出: (16, 2048)
    
    策略：2D Grid=(M, cdiv(N, BLOCK_N))，核内循环切分SUB_N
    """
    pid_m = tl.program_id(0)  # 行索引 [0, M)
    pid_n = tl.program_id(1)  # 列块索引
    
    # 计算当前核负责的列范围
    n_block_start = pid_n * BLOCK_N
    n_block_end = tl.minimum(n_block_start + BLOCK_N, N)
    
    # 核内循环：每次处理 SUB_N 列
    for sub_start in range(0, BLOCK_N, SUB_N):
        n_start = n_block_start + sub_start
        
        # 检查是否超出当前块的范围
        if n_start < n_block_end:
        
            # 计算列偏移
            offs_n = n_start + tl.arange(0, SUB_N)
            mask_n = offs_n < N
            
            # 加载 input2 的当前列块（广播的部分）
            input2 = tl.load(input2_ptr + offs_n, mask=mask_n, other=0.0)  # shape: (SUB_N,)
            
            # 计算 input1 的偏移（当前行和当前列块）
            input1_offs = pid_m * stride_in1_m + offs_n
            
            # 加载 input1
            input1 = tl.load(input1_ptr + input1_offs, mask=mask_n, other=0.0)
            
            # 计算：input2 自动广播
            output = input1 + input2
            
            # 存储
            output_offs = pid_m * stride_out_m + offs_n
            tl.store(output_ptr + output_offs, output, mask=mask_n)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    assert input1.ndim == 2 and input2.ndim == 2, "Both inputs must be 2D"
    assert input2.shape[0] == 1, "input2 must have shape (1, N)"
    assert input1.shape[1] == input2.shape[1], "Column dimension must match"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)
    
    M, N = input1.shape  # 16, 2048

    grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']))
    
    add_broadcast_kernel[grid](
        input1, input2, output,
        M, N,
        input1.stride(0), input2.stride(1), output.stride(0),
    )
    
    return output