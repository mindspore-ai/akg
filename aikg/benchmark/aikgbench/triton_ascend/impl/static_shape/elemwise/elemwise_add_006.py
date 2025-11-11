import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数，SUB_H 和 SUB_W 控制H和W维度的切分
        triton.Config({'NUM_BLOCKS': 32, 'SUB_H': 128, 'SUB_W': 256}), # 最优，核数被shape整除
        triton.Config({'NUM_BLOCKS': 32, 'SUB_H': 128, 'SUB_W': 128}), # ub没用满
        triton.Config({'NUM_BLOCKS': 32, 'SUB_H': 64, 'SUB_W': 256}), # ub没用满
        triton.Config({'NUM_BLOCKS': 40, 'SUB_H': 128, 'SUB_W': 256}), # 核数用满，但不能被shape整除
    ],
    key=['B', 'H', 'W'],
)
@triton.jit
def custom_op_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, H, W,
    stride_input1_b, stride_input1_h,
    stride_input2_h,
    stride_output_b, stride_output_h,
    NUM_BLOCKS: tl.constexpr,
    SUB_H: tl.constexpr,
    SUB_W: tl.constexpr,
):
    """
    优化的Triton kernel for 3D逐元素加法with广播
    输入形状: input1=(B, H, W), input2=(1, H, 1)
    输出形状: output=(B, H, W)
    
    策略：grid固定为NUM_BLOCKS，每个kernel处理分配的batch，内部切分H和W
    """
    pid = tl.program_id(0)
    
    # 每个block负责处理若干个batch
    batches_per_block = (B + NUM_BLOCKS - 1) // NUM_BLOCKS
    b_start = pid * batches_per_block
    b_end = tl.minimum(b_start + batches_per_block, B)
    
    # 外层循环：逐个处理batch
    for curr_b in range(b_start, b_end):
        # 中层循环：切分H维度
        for h_block in range(0, H, SUB_H):
            h_offs = h_block + tl.arange(0, SUB_H)
            mask_h = h_offs < H
            
            # 加载当前H块的input2值: (SUB_H,)
            input2_offs = h_offs * stride_input2_h
            input2_vals = tl.load(input2_ptr + input2_offs, mask=mask_h, other=0.0)
            
            # 内层循环：切分W维度
            for w_block in range(0, W, SUB_W):
                w_offs = w_block + tl.arange(0, SUB_W)
                mask_w = w_offs < W
                
                # 2D mask: (SUB_H, SUB_W)
                mask_2d = mask_h[:, None] & mask_w[None, :]
                
                # 计算input1的地址: (SUB_H, SUB_W)
                input1_offs = (curr_b * stride_input1_b + 
                              h_offs[:, None] * stride_input1_h + 
                              w_offs[None, :])
                
                # 计算output的地址
                output_offs = (curr_b * stride_output_b + 
                              h_offs[:, None] * stride_output_h + 
                              w_offs[None, :])
                
                # 加载input1: (SUB_H, SUB_W)
                input1 = tl.load(input1_ptr + input1_offs, mask=mask_2d, other=0.0)
                
                # 计算 - input2_vals广播到W维度
                output = input1 + input2_vals[:, None]
                
                # 存储
                tl.store(output_ptr + output_offs, output, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Triton实现for (B, H, W) + (1, H, 1) 广播
    grid固定为NUM_BLOCKS，内部切分B、H和W维度
    """
    assert input1.ndim == 3 and input2.ndim == 3, "Both inputs must be 3D"
    assert input2.shape[0] == 1 and input2.shape[2] == 1, "input2 must be (1, H, 1)"
    assert input1.shape[1] == input2.shape[1], "H dimension must match"
    assert input1.dtype == input2.dtype, "Dtypes must match"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    output = torch.empty_like(input1)
    
    B, H, W = input1.shape  # 64, 128, 256
    
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    custom_op_kernel[grid](
        input1, input2, output,
        B, H, W,
        input1.stride(0), input1.stride(1),
        input2.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output