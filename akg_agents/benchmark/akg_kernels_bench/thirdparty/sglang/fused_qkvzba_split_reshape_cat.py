import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/models/qwen3_next.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   fused_qkvzba_split_reshape_cat(mixed_qkvz, mixed_ba, num_heads_qk,
#                                   num_heads_v, head_qk, head_v)
# Triton Kernel:
#   fused_qkvzba_split_reshape_cat_kernel - Qwen3 GatedDeltaNet的QKV/Z/B/A张量分割重塑拼接
# ============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(v_end, z_end)
    )
    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        batch, seq_len = mixed_qkvz.shape[0], 1
        qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
        mixed_qkv = torch.empty(
            [batch * seq_len, qkv_dim_t],
            dtype=mixed_qkvz.dtype,
            device=mixed_qkvz.device,
        )
        z = torch.empty(
            [batch * seq_len, num_heads_v, head_v],
            dtype=mixed_qkvz.dtype,
            device=mixed_qkvz.device,
        )
        b = torch.empty(
            [batch * seq_len, num_heads_v],
            dtype=mixed_ba.dtype,
            device=mixed_ba.device,
        )
        a = torch.empty_like(b)
        grid = (batch * seq_len, num_heads_qk)
        fused_qkvzba_split_reshape_cat_kernel[grid](
            mixed_qkv,
            z,
            b,
            a,
            mixed_qkvz,
            mixed_ba,
            num_heads_qk,
            num_heads_v,
            head_qk,
            head_v,
            num_warps=1,
            num_stages=3,
        )
        return mixed_qkv, z, b, a


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        from sglang.srt.models.qwen3_next import fused_qkvzba_split_reshape_cat
        return fused_qkvzba_split_reshape_cat(
            mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
        )


def get_inputs():
    # Example dimensions for Qwen3 GatedDeltaNet
    batch = 8
    num_heads_qk = 8
    num_heads_v = 16
    head_qk = 64
    head_v = 64
    dtype = torch.float16

    qkvz_dim = num_heads_qk * (head_qk * 2 + num_heads_v // num_heads_qk * head_v * 2)
    ba_dim = num_heads_qk * (num_heads_v // num_heads_qk * 2)

    mixed_qkvz = torch.randn(batch, qkvz_dim, dtype=dtype)
    mixed_ba = torch.randn(batch, ba_dim, dtype=dtype)

    return [mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v]


def get_init_inputs():
    return []
