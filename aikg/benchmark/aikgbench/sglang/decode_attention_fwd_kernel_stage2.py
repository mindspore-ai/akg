import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# 测试文件: test/srt/test_triton_attention_kernels.py (test_decode_attention)
# SGLang API调用:
#   _decode_softmax_reducev_fwd(logits, lse, q, o, v_buffer, kv_indptr, num_kv_splits, max_kv_splits)
# Triton Kernel:
#   _fwd_kernel_stage2 - decode attention stage2 (reduce)
# ============================================================================

_MIN_BLOCK_KV = 32


@triton.jit
def _fwd_kernel_stage2(
    Mid_O, Mid_O_1, O, kv_indptr, num_kv_splits, sink_ptr,
    stride_mid_ob, stride_mid_oh, stride_mid_os, stride_obs, stride_oh,
    MAX_KV_SPLITS: tl.constexpr, MIN_BLOCK_KV: tl.constexpr,
    BLOCK_DV: tl.constexpr, Lv: tl.constexpr, HAS_SINK: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    kv_splits = tl.load(num_kv_splits + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh) // Lv
    kv_len_per_split = tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV

    for split_kv_id in range(0, MAX_KV_SPLITS):
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O_1 + offs_logic + split_kv_id * stride_mid_os // Lv)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    if HAS_SINK:
        cur_sink = tl.load(sink_ptr + cur_head)
        e_sum += tl.exp(cur_sink - e_max)

    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / e_sum, mask=mask_d)


def _decode_softmax_reducev_fwd_impl(logits, lse, q, o, v_buffer, kv_indptr, num_kv_splits, max_kv_splits, sinks=None):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    MAX_KV_SPLITS = max_kv_splits
    HAS_SINK = sinks is not None

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits, lse, o, kv_indptr, num_kv_splits, sinks,
        logits.stride(0), logits.stride(1), logits.stride(2),
        o.stride(0), o.stride(1),
        MAX_KV_SPLITS=MAX_KV_SPLITS, MIN_BLOCK_KV=_MIN_BLOCK_KV,
        BLOCK_DV=BLOCK_DV, Lv=Lv, HAS_SINK=HAS_SINK,
    )


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, att_out, att_lse, q, v_buffer, kv_indptr, num_kv_splits):
        batch, num_heads, max_kv_splits, head_dim = att_out.shape
        o = torch.empty(batch, num_heads, head_dim, dtype=att_out.dtype, device=att_out.device)
        _decode_softmax_reducev_fwd_impl(att_out, att_lse, q, o, v_buffer, kv_indptr, num_kv_splits, max_kv_splits)
        return o


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self):
        super(ModelSglang, self).__init__()

    def forward(self, att_out, att_lse, q, v_buffer, kv_indptr, num_kv_splits):
        """
        调用SGLang _decode_softmax_reducev_fwd API
        """
        from sglang.srt.layers.attention.triton_ops.decode_attention import _decode_softmax_reducev_fwd

        batch, num_heads, max_kv_splits, head_dim = att_out.shape
        o = torch.empty(batch, num_heads, head_dim, dtype=att_out.dtype, device=att_out.device)
        _decode_softmax_reducev_fwd(att_out, att_lse, q, o, v_buffer, kv_indptr, num_kv_splits, max_kv_splits)
        return o


def get_inputs():
    batch_size = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    max_kv_splits = 8
    max_seq_len = 512
    dtype = torch.float32
    att_out = torch.randn(batch_size, num_heads, max_kv_splits, head_dim, dtype=dtype)
    att_lse = torch.randn(batch_size, num_heads, max_kv_splits, dtype=dtype)
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    seq_lens = torch.randint(64, max_seq_len, (batch_size,))
    total_kv_tokens = seq_lens.sum().item()
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    num_kv_splits = torch.full((batch_size,), max_kv_splits, dtype=torch.int32)
    return [att_out, att_lse, q, v_buffer, kv_indptr, num_kv_splits]


def get_init_inputs():
    return []
