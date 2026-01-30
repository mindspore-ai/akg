import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Grid切分B和H维度，每个kernel处理(BLOCK_B, BLOCK_H)的2D块
        # Grid = (cdiv(B, BLOCK_B), cdiv(H, BLOCK_H))
        # 优化：减少地址计算，复用input_offs
        triton.Config({'BLOCK_B': 16, 'BLOCK_H': 10}),   # Grid=(8, 5) = 40核, 最优
        triton.Config({'BLOCK_B': 8, 'BLOCK_H': 10}),    # Grid=(16, 5) = 80核, 核数>40，调度开销大
        triton.Config({'BLOCK_B': 8, 'BLOCK_H': 25}),    # Grid=(16, 2) = 32核, 核数<40，未用满
        triton.Config({'BLOCK_B': 4, 'BLOCK_H': 25}),    # Grid=(32, 2) = 64核
        triton.Config({'BLOCK_B': 32, 'BLOCK_H': 5}),    # Grid=(4, 10) = 40核
        triton.Config({'BLOCK_B': 16, 'BLOCK_H': 5}),    # Grid=(8, 10) = 80核
    ],
    key=['B', 'H'],
)
@triton.jit
def concat_slice_kernel(
    # 7个输入指针
    x1_ptr, x2_ptr, x3_ptr, x4_ptr, x5_ptr, x6_ptr, x7_ptr,
    output_ptr,
    B, H,
    # strides (假设所有输入stride相同)
    stride_in_b, stride_in_h, stride_in_w,
    stride_out_b, stride_out_h, stride_out_w,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    # 切片大小 (必须是 constexpr)
    SLICE_1: tl.constexpr,  # 128
    SLICE_2: tl.constexpr,  # 32
    SLICE_3: tl.constexpr,  # 48
):
    """
    Concat with slice kernel - 2D Grid切分B和H
    输入: 7个 (128, 50, 128)
    切片: [128, 32, 48, 48, 48, 48, 48]
    输出: (128, 50, 400)
    
    策略：Grid=(cdiv(B, BLOCK_B), cdiv(H, BLOCK_H))，一次处理(BLOCK_B, BLOCK_H)的2D块
    """
    pid_b = tl.program_id(0)  # B维度的block索引
    pid_h = tl.program_id(1)  # H维度的block索引
    
    # 计算B和H的偏移
    b_start = pid_b * BLOCK_B
    h_start = pid_h * BLOCK_H
    
    b_offs = b_start + tl.arange(0, BLOCK_B)  # (BLOCK_B,)
    h_offs = h_start + tl.arange(0, BLOCK_H)  # (BLOCK_H,)
    
    mask_b = b_offs < B
    mask_h = h_offs < H
    mask_bh = mask_b[:, None] & mask_h[None, :]  # (BLOCK_B, BLOCK_H)
    
    # 预计算B和H的地址基础部分（复用7次）
    base_in_offs = b_offs[:, None, None] * stride_in_b + h_offs[None, :, None] * stride_in_h
    base_out_offs = b_offs[:, None, None] * stride_out_b + h_offs[None, :, None] * stride_out_h
    
    # 当前输出的W偏移
    w_out_offset = 0
    
    # ========== Input 1: 切片[:SLICE_1] → 输出[0:SLICE_1] ==========
    w_offs_1 = tl.arange(0, SLICE_1)
    mask_1 = mask_bh[:, :, None]  # W维度总是有效的（SLICE_1=128 < W_in=128）
    
    input_offs = base_in_offs + w_offs_1[None, None, :] * stride_in_w
    output_offs = base_out_offs + (w_out_offset + w_offs_1)[None, None, :] * stride_out_w
    
    data = tl.load(x1_ptr + input_offs, mask=mask_1, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_1)
    w_out_offset += SLICE_1
    
    # ========== Input 2: 切片[:SLICE_2] ==========
    w_offs_2 = tl.arange(0, SLICE_2)
    mask_2 = mask_bh[:, :, None]
    
    input_offs = base_in_offs + w_offs_2[None, None, :] * stride_in_w
    output_offs = base_out_offs + (w_out_offset + w_offs_2)[None, None, :] * stride_out_w
    
    data = tl.load(x2_ptr + input_offs, mask=mask_2, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_2)
    w_out_offset += SLICE_2
    
    # ========== Input 3-7: 切片[:SLICE_3]，使用相同的w_offs和input_offs基础 ==========
    w_offs_3 = tl.arange(0, SLICE_3)
    mask_3 = mask_bh[:, :, None]
    input_offs_base = base_in_offs + w_offs_3[None, None, :] * stride_in_w
    
    # Input 3
    output_offs = base_out_offs + (w_out_offset + w_offs_3)[None, None, :] * stride_out_w
    data = tl.load(x3_ptr + input_offs_base, mask=mask_3, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_3)
    w_out_offset += SLICE_3
    
    # Input 4
    output_offs = base_out_offs + (w_out_offset + w_offs_3)[None, None, :] * stride_out_w
    data = tl.load(x4_ptr + input_offs_base, mask=mask_3, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_3)
    w_out_offset += SLICE_3
    
    # Input 5
    output_offs = base_out_offs + (w_out_offset + w_offs_3)[None, None, :] * stride_out_w
    data = tl.load(x5_ptr + input_offs_base, mask=mask_3, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_3)
    w_out_offset += SLICE_3
    
    # Input 6
    output_offs = base_out_offs + (w_out_offset + w_offs_3)[None, None, :] * stride_out_w
    data = tl.load(x6_ptr + input_offs_base, mask=mask_3, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_3)
    w_out_offset += SLICE_3
    
    # Input 7
    output_offs = base_out_offs + (w_out_offset + w_offs_3)[None, None, :] * stride_out_w
    data = tl.load(x7_ptr + input_offs_base, mask=mask_3, other=0.0)
    tl.store(output_ptr + output_offs, data, mask=mask_3)


def custom_op_triton_torch(*xs):
    """
    Triton实现: 7个输入切片后拼接 (优化版)
    输入: 7个 (128, 50, 128)
    切片: [128, 32, 48, 48, 48, 48, 48]
    输出: (128, 50, 400)
    
    优化：2D Grid切分B和H维度，向量化处理
    """
    assert len(xs) == 7, f"Expected 7 inputs, got {len(xs)}"
    
    # 验证形状和类型
    expected_shape = (128, 50, 128)
    for i, x in enumerate(xs):
        assert x.shape == expected_shape, f"Input {i} shape mismatch: {x.shape} vs {expected_shape}"
        assert x.dtype == xs[0].dtype, f"Input {i} dtype mismatch"
    
    # 确保连续
    xs = [x.contiguous() for x in xs]
    
    B, H, W_in = xs[0].shape  # (128, 50, 128)
    W_out = 128 + 32 + 48 + 48 + 48 + 48 + 48  # 400
    
    # 创建输出张量
    output = torch.empty((B, H, W_out), dtype=xs[0].dtype, device=xs[0].device)
    
    # Grid 2D: (cdiv(B, BLOCK_B), cdiv(H, BLOCK_H))
    grid = lambda meta: (
        triton.cdiv(B, meta['BLOCK_B']),
        triton.cdiv(H, meta['BLOCK_H']),
    )
    
    # 解包7个输入传给kernel
    concat_slice_kernel[grid](
        xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6],
        output,
        B, H,
        xs[0].stride(0), xs[0].stride(1), xs[0].stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        SLICE_1=128,
        SLICE_2=32,
        SLICE_3=48,
    )
    
    return output