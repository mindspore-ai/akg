import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_B': 16, 'BLOCK_H': 8}), # 性能较差
        triton.Config({'BLOCK_B': 8, 'BLOCK_H': 16}), # 最优，核数等于 4*8=32
        triton.Config({'BLOCK_B': 4, 'BLOCK_H': 32}), # 性能非常接近最优
    ],
    key=['B', 'H'],  # 根据 B 和 H 自动选择最优配置
)


@triton.jit
def custom_op_kernel(
    a_ptr, y_ptr, c_ptr,
    B, H, W,
    stride_a_b, stride_a_h, stride_a_w,
    stride_y_h, stride_y_w,
    stride_c_b, stride_c_h, stride_c_w,
    BLOCK_B: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # program_id(0) -> which batch-block
    # program_id(1) -> which h-block
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    b_start = pid_b * BLOCK_B
    h_start = pid_h * BLOCK_H

    b_offsets = b_start + tl.arange(0, BLOCK_B)     # (BLOCK_B,)
    h_offsets = h_start + tl.arange(0, BLOCK_H)     # (BLOCK_H,)
    w_offsets = tl.arange(0, BLOCK_W)               # (BLOCK_W,)

    mask_b = b_offsets < B
    mask_h = h_offsets < H
    mask_w = w_offsets < W
    # mask shape: (BLOCK_B, BLOCK_H, BLOCK_W)
    mask = mask_b[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]

    # compute offsets for a and c (depend on b,h,w)
    a_off = (
        b_offsets[:, None, None] * stride_a_b +
        h_offsets[None, :, None] * stride_a_h +
        w_offsets[None, None, :] * stride_a_w
    )
    c_off = (
        b_offsets[:, None, None] * stride_c_b +
        h_offsets[None, :, None] * stride_c_h +
        w_offsets[None, None, :] * stride_c_w
    )

    # y has shape (1, H, W) and should NOT depend on b (batch broadcast).
    # compute y offsets based only on h and w
        # --- 计算 offsets（确保 y_off 是二维 (BLOCK_H, BLOCK_W)） ---
    # 注意：h_offsets shape (BLOCK_H,), w_offsets shape (BLOCK_W,)
    y_off = h_offsets[:, None] * stride_y_h + w_offsets[None, :] * stride_y_w
    # y_off shape -> (BLOCK_H, BLOCK_W)

    # --- 载入 a （形状: BLOCK_B x BLOCK_H x BLOCK_W） ---
    a = tl.load(a_ptr + a_off, mask=mask, other=0.0)

    # --- 载入 y（只依赖 h,w -> 形状: BLOCK_H x BLOCK_W） ---
    y_mask = mask_h[:, None] & mask_w[None, :]

    y_val = tl.load(y_ptr + y_off, mask=y_mask, other=0.0)  # shape (BLOCK_H, BLOCK_W)

    # --- 广播 y 到 batch 维，再做加法（a: BLOCK_B x BLOCK_H x BLOCK_W） ---
    c = a + y_val[None, :, :]  # y_val[None,...] -> (1, BLOCK_H, BLOCK_W) -> broadcasts to BLOCK_B
    tl.store(c_ptr + c_off, c, mask=mask)



def custom_op_triton_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation for (B, H, W) + (1, H, W),
    where only the batch dimension is broadcast (y.shape[0] == 1).
    We slice/parallelize over first two dims (B, H).
    """
    assert x.ndim == 3 and y.ndim == 3
    assert y.shape[0] == 1 and y.shape[1] == x.shape[1] and y.shape[2] == x.shape[2],         "y must be (1, H, W) to broadcast only in batch"

    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)

    B, H, W = x.shape

    # block settings: tune as needed
    # BLOCK_B = 4     # batches per block
    # BLOCK_H = 32    # heights per block
    BLOCK_W = 256  # process up to 256 columns per block (tune for your GPU)

    grid = lambda META: (
        triton.cdiv(B, META['BLOCK_B']),
        triton.cdiv(H, META['BLOCK_H']),
    )

    custom_op_kernel[grid](
        x, y, out,
        B, H, W,
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(1), y.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_W=BLOCK_W,
    )
    return out