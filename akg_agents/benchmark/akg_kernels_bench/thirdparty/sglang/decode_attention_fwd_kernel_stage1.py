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
#   decode_attention_fwd(q, k_buffer, v_buffer, o, kv_indptr, kv_indices, ...)
# Triton Kernel:
#   _fwd_kernel_stage1 - decode attention stage1
# ============================================================================


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


_is_hip = is_hip()
_MIN_BLOCK_KV = 32


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    kv_indptr,
    kv_indices,
    Att_Out,
    Att_Lse,
    num_kv_splits,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_BLOCK_KV: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
    xai_temperature_len: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = tl.load(num_kv_splits + cur_batch)

    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / tl.log2(float(xai_temperature_len))
        _qtemp = tl.log2(offs_qidx.to(tl.float32)) * xai_temperature_scale
        xai_temperature_reg = tl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d

    kv_len_per_split = (
        tl.cdiv(tl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    )
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        q = tl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            if xai_temperature_len > 0:
                qk *= xai_temperature_reg

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
        ) // Lv

        tl.store(
            Att_Lse + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd_impl(
    q,
    k_buffer,
    v_buffer,
    att_out,
    att_lse,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    max_kv_splits,
    sm_scale,
    logit_cap,
    xai_temperature_len=-1,
):
    BLOCK = 64
    if _is_hip:
        BLOCK = 8
    MAX_KV_SPLITS = max_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = q.shape[0], q.shape[1]

    grid = (batch, head_num, MAX_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if _is_hip:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        att_lse,
        num_kv_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        MIN_BLOCK_KV=_MIN_BLOCK_KV,
        logit_cap=logit_cap,
        xai_temperature_len=xai_temperature_len,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self, sm_scale=None, logit_cap=0.0):
        super(Model, self).__init__()
        self.sm_scale = sm_scale
        self.logit_cap = logit_cap

    def forward(self, q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits):
        batch, num_heads, head_dim = q.shape
        max_kv_splits = num_kv_splits.max().item()
        sm_scale = self.sm_scale if self.sm_scale else 1.0 / (head_dim ** 0.5)
        Lv = v_buffer.shape[-1]

        att_out = torch.empty(batch, num_heads, max_kv_splits, Lv, dtype=torch.float32, device=q.device)
        att_lse = torch.empty(batch, num_heads, max_kv_splits, dtype=torch.float32, device=q.device)

        _decode_att_m_fwd_impl(
            q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices,
            num_kv_splits, max_kv_splits, sm_scale, self.logit_cap
        )
        return att_out, att_lse


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self, sm_scale=None, logit_cap=0.0):
        super(ModelSglang, self).__init__()
        self.sm_scale = sm_scale
        self.logit_cap = logit_cap

    def forward(self, q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits):
        from sglang.srt.layers.attention.triton_ops.decode_attention import _decode_att_m_fwd

        batch, num_heads, head_dim = q.shape
        max_kv_splits = num_kv_splits.max().item()
        sm_scale = self.sm_scale if self.sm_scale else 1.0 / (head_dim ** 0.5)
        Lv = v_buffer.shape[-1]

        att_out = torch.empty(batch, num_heads, max_kv_splits, Lv, dtype=torch.float32, device=q.device)
        att_lse = torch.empty(batch, num_heads, max_kv_splits, dtype=torch.float32, device=q.device)

        _decode_att_m_fwd(
            q, k_buffer, v_buffer, att_out, att_lse, kv_indptr, kv_indices,
            num_kv_splits, max_kv_splits, sm_scale, self.logit_cap
        )
        return att_out, att_lse


def get_inputs():
    batch_size = 4
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    max_seq_len = 512
    max_kv_splits = 8
    dtype = torch.float16
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype)
    seq_lens = torch.randint(64, max_seq_len, (batch_size,))
    total_kv_tokens = seq_lens.sum().item()
    k_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    v_buffer = torch.randn(total_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.arange(total_kv_tokens, dtype=torch.int32)
    num_kv_splits = torch.full((batch_size,), max_kv_splits, dtype=torch.int32)
    return [q, k_buffer, v_buffer, kv_indptr, kv_indices, num_kv_splits]


def get_init_inputs():
    return [None, 0.0]
