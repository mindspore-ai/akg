import triton
import triton.language as tl
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    mask_block_ptr,
    stride_k_seqlen,
    stride_v_seqlen,
    stride_attn_mask_kv_seqlen,
    start_m,
    qk_scale,
    q_load_mask,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    KV_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
    else:
        lo, hi = 0, KV_CTX

    K_block_ptr += lo * stride_k_seqlen
    V_block_ptr += lo * stride_v_seqlen
    if HAS_ATTN_MASK:
        mask_block_ptr += lo * stride_attn_mask_kv_seqlen

    LOG2E: tl.constexpr = 1.44269504

    for start_n in range(lo, hi, BLOCK_N):
        kv_load_mask = (start_n + offs_n) < KV_CTX
        k = tl.load(K_block_ptr, mask=kv_load_mask[None, :], other=0.0)
        if PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)

        qk = tl.dot(q, k, allow_tf32=False)
        qk = tl.where(kv_load_mask[None, :], qk, -float("inf"))

        if HAS_ATTN_MASK:
            attn_mask = tl.load(
                mask_block_ptr,
                mask=q_load_mask[:, None] & kv_load_mask[None, :],
                other=0.0,
            )

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            if HAS_ATTN_MASK:
                qk = qk * qk_scale + attn_mask
                qk *= LOG2E
                qk = qk + tl.where(mask, 0, -1.0e6)
            else:
                qk = qk * qk_scale * LOG2E + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk *= qk_scale * LOG2E
            if HAS_ATTN_MASK:
                qk = qk + attn_mask
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(V_block_ptr, mask=kv_load_mask[:, None], other=0.0)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(q.dtype)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc, allow_tf32=False)
        m_i = m_ij

        K_block_ptr += BLOCK_N * stride_k_seqlen
        V_block_ptr += BLOCK_N * stride_v_seqlen
        if HAS_ATTN_MASK:
            mask_block_ptr += BLOCK_N * stride_attn_mask_kv_seqlen

    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=8),
    ],
    key=["KV_CTX", "HEAD_DIM"],
)
@triton.jit
def flash_attention_kernel(
    Q,
    K,
    V,
    attn_mask,
    sm_scale,
    Out,
    stride_q_batch,
    stride_q_head,
    stride_q_seqlen,
    stride_q_headsize,
    stride_k_batch,
    stride_k_head,
    stride_k_seqlen,
    stride_k_headsize,
    stride_v_batch,
    stride_v_head,
    stride_v_seqlen,
    stride_v_headsize,
    stride_attn_mask_batch,
    stride_attn_mask_head,
    stride_attn_mask_q_seqlen,
    stride_attn_mask_kv_seqlen,
    stride_o_batch,
    stride_o_head,
    stride_o_seqlen,
    stride_o_headsize,
    Z,
    q_numhead,
    kv_numhead,
    Q_CTX,
    KV_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    HAS_ATTN_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // q_numhead
    head_id = off_hz % q_numhead
    kv_head_id = off_hz % kv_numhead

    q_offset = batch_id * stride_q_batch + head_id * stride_q_head
    o_offset = batch_id * stride_o_batch + head_id * stride_o_head
    kv_offset = batch_id * stride_k_batch + kv_head_id * stride_k_head

    offs_headsize = tl.arange(0, HEAD_DIM)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    Q_block_ptr = Q + q_offset + offs_m[:, None] * stride_q_seqlen + offs_headsize[None, :] * stride_q_headsize
    K_block_ptr = K + kv_offset + offs_n[None, :] * stride_k_seqlen + offs_headsize[:, None] * stride_k_headsize
    V_block_ptr = V + kv_offset + offs_n[:, None] * stride_v_seqlen + offs_headsize[None, :] * stride_v_headsize

    if HAS_ATTN_MASK:
        attn_mask_offset = batch_id * stride_attn_mask_batch + head_id * stride_attn_mask_head
        mask_block_ptr = attn_mask + attn_mask_offset + offs_m[:, None] * stride_attn_mask_q_seqlen + offs_n[None, :] * stride_attn_mask_kv_seqlen
    else:
        mask_block_ptr = None

    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_o_seqlen + offs_headsize[None, :] * stride_o_headsize

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q_load_mask = offs_m < Q_CTX
    q = tl.load(Q_block_ptr, mask=q_load_mask[:, None], other=0.0)

    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            sm_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )

    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask_block_ptr,
            stride_k_seqlen,
            stride_v_seqlen,
            stride_attn_mask_kv_seqlen,
            start_m,
            sm_scale,
            q_load_mask,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            KV_CTX,
            V.dtype.element_ty == tl.float8e5,
            HAS_ATTN_MASK,
            PRE_LOAD_V,
        )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=q_load_mask[:, None])


class ModelNew(torch.nn.Module):
    def __init__(self, dropout_p=0.0, causal=False, softmax_scale=None):
        super().__init__()
        # 硬编码参数，根据任务描述
        self.dropout_p = dropout_p
        self.causal = causal
        self.softmax_scale = softmax_scale
        # 根据任务描述硬编码dropout_p为0.0
        assert dropout_p == 0.0, "Currenty only support dropout_p=0.0"

    def forward(self, q, k, v, attn_mask=None):
        # 从输入张量获取shape参数
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]
        kv_numhead = k.shape[2]  # 假设k和v的num_heads相同，即MHA

        # 调整输入形状以适配kernel: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        # 硬编码的参数：设置scale factor
        if self.softmax_scale is None:
            sm_scale = 1.0 / (head_dim ** 0.5)
        else:
            sm_scale = self.softmax_scale

        o = torch.empty_like(q, dtype=v.dtype)

        # 硬编码的参数：根据causal标志设置stage
        stage = 3 if self.causal else 1
        
        # 处理注意力掩码
        if attn_mask is not None:
            HAS_ATTN_MASK = True
            PRE_LOAD_V = False
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.to(q.dtype) * -1.0e6
            # 调整attn_mask形状以适配kernel: 需要是4维 [batch, num_heads, seq_len_q, seq_len_kv]
            if attn_mask.dim() == 2:
                # (seq_len_q, seq_len_kv) -> (batch, num_heads, seq_len_q, seq_len_kv)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len_q, seq_len_kv)
            elif attn_mask.dim() == 3:
                # (batch, seq_len_q, seq_len_kv) -> (batch, num_heads, seq_len_q, seq_len_kv)
                attn_mask = attn_mask.unsqueeze(1).expand(batch_size, num_heads, seq_len_q, seq_len_kv)
            # 调整掩码的维度以匹配转置后的qkv布局 [batch, num_heads, seq_len, seq_len]
            attn_mask = attn_mask.transpose(1, 2).contiguous()
            mask_strides = attn_mask.stride()
        else:
            HAS_ATTN_MASK = False
            PRE_LOAD_V = False
            mask_strides = (1, 1, 1, 1)
            attn_mask = torch.empty(0, device=q.device, dtype=q.dtype)

        # 定义网格函数，动态计算网格大小
        grid = lambda meta: (
            triton.cdiv(seq_len_q, meta['BLOCK_M']),
            batch_size * num_heads,
            1,
        )

        # 调用内核函数
        flash_attention_kernel[grid](
            q,
            k,
            v,
            attn_mask if HAS_ATTN_MASK else None,
            sm_scale,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            mask_strides[3],
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            batch_size,
            num_heads,
            kv_numhead,
            seq_len_q,
            seq_len_kv,
            HEAD_DIM=head_dim,
            STAGE=stage,
            HAS_ATTN_MASK=HAS_ATTN_MASK,
            PRE_LOAD_V=PRE_LOAD_V,
        )

        # 调整输出形状回原始格式: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        o = o.transpose(1, 2).contiguous()
        return o