import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        # NUM_BLOCKS 核数，SUB_B和SUB_H分别控制B和H维度的切分
        triton.Config({'NUM_BLOCKS': 32, 'SUB_B': 2, 'SUB_H': 64}), # 最优，核数被shape整除
        triton.Config({'NUM_BLOCKS': 32, 'SUB_B': 4, 'SUB_H': 32}), # H偏小
        triton.Config({'NUM_BLOCKS': 64, 'SUB_B': 2, 'SUB_H': 64}), # 核数 > 40
        triton.Config({'NUM_BLOCKS': 40, 'SUB_B': 2, 'SUB_H': 64}), # shape不能被核数整除

    ],
    key=['B', 'H', 'W'],
)

@triton.jit
def custom_op_kernel(
    a_ptr, y_ptr, c_ptr,
    B, H,
    W: tl.constexpr,
    stride_a_b, stride_a_h,
    stride_y_b,
    stride_c_b, stride_c_h,
    NUM_BLOCKS: tl.constexpr,
    SUB_B: tl.constexpr,
    SUB_H: tl.constexpr,
):
    """
    优化的Triton kernel for 3D逐元素乘法with广播
    输入形状: a=(B, H, W), y=(B, 1, W)
    输出形状: c=(B, H, W)
    
    策略：两层循环分别切B和H维度，W维度完整处理
    外层循环遍历B（每次SUB_B个batch），内层循环遍历H（每次SUB_H行）
    """
    pid = tl.program_id(0)
    
    # 每个block负责处理一部分batch
    batches_per_block = (B + NUM_BLOCKS - 1) // NUM_BLOCKS
    b_start = pid * batches_per_block
    b_end = tl.minimum(b_start + batches_per_block, B)
    
    # W 维度的偏移（完整处理W维度）
    offs_w = tl.arange(0, W)
    
    # 外层循环：遍历B维度，每次处理SUB_B个batch
    for b_block in range(b_start, b_end, SUB_B):
        # 内层循环：遍历H维度，每次处理SUB_H行
        for h_block in range(0, H, SUB_H):
            # 当前处理的H索引
            h_offs = h_block + tl.arange(0, SUB_H)
            mask_h = h_offs < H
            
            # 遍历SUB_B个batch
            for sub_b_idx in range(SUB_B):
                curr_b = b_block + sub_b_idx
                
                if curr_b < b_end:
                    # 2D 索引: (SUB_H, W)
                    h_offs_2d = h_offs[:, None]
                    offs_w_2d = offs_w[None, :]

                    mask_2d = mask_h[:, None]
                    a_offs = curr_b * stride_a_b + h_offs_2d * stride_a_h + offs_w_2d
                    y_offs = curr_b * stride_y_b + offs_w
                    c_offs = curr_b * stride_c_b + h_offs_2d * stride_c_h + offs_w_2d

                    a = tl.load(a_ptr + a_offs, mask=mask_2d, other=0.0)
                    y = tl.load(y_ptr + y_offs)
                    c = a * y

                    tl.store(c_ptr + c_offs, c, mask=mask_2d)


def custom_op_triton_torch(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """
    Triton实现for (B, H, W) * (B, 1, W) 广播
    将(B, H)展平切分，W维度完整处理（保持连续性）
    参考div001的写法，避免256*256超出Triton限制
    """
    assert input1.ndim == 3 and input2.ndim == 3, "Both inputs must be 3D"
    assert input2.shape[1] == 1, "input2 dim1 must be 1 for broadcasting"
    assert input1.shape[0] == input2.shape[0], "Batch size must match"
    assert input1.shape[2] == input2.shape[2], "W dimension must match"
    assert input1.dtype == input2.dtype, "Dtypes must match"
    
    input1 = input1.contiguous()
    input2 = input2.contiguous()
    out = torch.empty_like(input1)
    
    B, H, W = input1.shape  # 128, 256, 256
    
    # Grid 固定为 NUM_BLOCKS，匹配硬件核数
    grid = lambda meta: (meta['NUM_BLOCKS'],)
    
    custom_op_kernel[grid](
        input1, input2, out,
        B, H, W,
        input1.stride(0), input1.stride(1),
        input2.stride(0),
        out.stride(0), out.stride(1),
    )
    return out