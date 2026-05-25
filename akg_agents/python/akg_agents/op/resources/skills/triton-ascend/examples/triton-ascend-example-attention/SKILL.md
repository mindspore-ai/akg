---
name: triton-ascend-example-attention
description: "A5（Ascend950）Flash Attention 的完整 Triton-Ascend 实现示例。Cube/Vector 串行交替：Cube 跑 logits = Q·K^T 与 partial = P·V 两段 matmul + fixpipe；Vector 跑在线 softmax + flash 累加更新；通过 3 个 sync_block 事件（id 0/1/2，PIPE_FIX/V/MTE 配对）协调。可参考此代码结构生成 A5 串行 attention 类算子。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  operator_type: "attention"
  requires_affinity: true
---

# A5 Flash Attention — 实现示例

```python
@triton.jit
def _cube_compute_logits(
    q_tile,
    k_blk,
    logits_ub,
    D_HEAD: tl.constexpr,
    BN: tl.constexpr,
):
    k_tile = tl.load(k_blk)
    s_tile = tl.dot(q_tile, tl.trans(k_tile))
    al.fixpipe(s_tile, logits_ub,
               al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)

@triton.jit
def _cube_apply_attn_to_value(
    prob_l1_nz,
    attn_value_ub,
    v_blk,
    D_HEAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    v_tile = tl.load(v_blk)
    p_view = bl.to_tensor(prob_l1_nz, target_shape=[BM, BN])
    o_partial = tl.dot(p_view, v_tile)
    al.fixpipe(o_partial, attn_value_ub,
               al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)

# Vector helper — flash-attention style accumulator update.
#   out_acc <- out_acc * rescale[:, None] + partial(P·V)

@triton.jit
def _vector_accumulate_partial(
    attn_value_ub,
    rescale,
    out_acc,
    D_HEAD: tl.constexpr,
    BM: tl.constexpr,
):
    o_partial = bl.to_tensor(attn_value_ub)
    out_acc = out_acc * rescale[:, None] + o_partial
    return out_acc

# Main kernel — flash attention forward, serial Cube/Vector cooperation.
@triton.jit
def flash_attn_fwd_serial_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    softmax_scale,
    s_q_b, s_q_h, s_q_m, s_q_d,    # Q strides: batch / head / m / d
    s_k_n, s_k_d,                  # K strides over (n_ctx, d_head) per (b,h)
    s_v_n, s_v_d,                  # V strides over (n_ctx, d_head) per (b,h)
    s_o_m, s_o_d,                  # Out strides (m, d)
    B, NHEAD, SEQ_LEN,
    NUM_CORES: tl.constexpr,
    D_HEAD: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(0)
    elem_ty = q_ptr.dtype.element_ty

    n_m_blocks = SEQ_LEN // BM
    n_total = B * NHEAD * n_m_blocks
    HALF_M: tl.constexpr = BM // 2

    # ---- on-chip buffers (UB rows = HALF_M for ROW_SPLIT; P 在 L1 用 NZ 16×16 分形) ----
    logits_ub     = bl.alloc(tl.float32, (HALF_M, BN),     al.ascend_address_space.UB)
    attn_value_ub = bl.alloc(tl.float32, (HALF_M, D_HEAD), al.ascend_address_space.UB)
    prob_l1_nz    = bl.alloc(elem_ty, (BN // 16, BM // 16, 16, 16),
                             al.ascend_address_space.L1)

    for tile_id in range(pid, n_total, NUM_CORES):
        # Decode (batch, head, m-block).
        bh_idx = tile_id // n_m_blocks
        mb_idx = tile_id %  n_m_blocks
        batch_id = bh_idx // NHEAD
        head_id  = bh_idx %  NHEAD
        bh_base_off = batch_id.to(tl.int64) * s_q_b + head_id.to(tl.int64) * s_q_h

        # q_blk / k_blk / v_blk: tl.make_block_ptr 标准构造，order=(1,0)
        #   都用 base = {q,k,v}_ptr + bh_base_off，shape = (SEQ_LEN, D_HEAD)
        #   q_blk: strides=(s_q_m, s_q_d), offsets=(mb_idx*BM, 0), block_shape=(BM, D_HEAD)
        #   k_blk: strides=(s_k_n, s_k_d), offsets=(0, 0),         block_shape=(BN, D_HEAD)
        #   v_blk: strides=(s_v_n, s_v_d), offsets=(0, 0),         block_shape=(BN, D_HEAD)

        # ---- online softmax running state (per sub-vector half) ----
        row_max     = tl.full([HALF_M],          -float("inf"), dtype=tl.float32)
        row_denom   = tl.full([HALF_M],          0,             dtype=tl.float32)
        out_acc     = tl.full([HALF_M, D_HEAD],  0,             dtype=tl.float32)
        rescale     = tl.full([HALF_M],          0,             dtype=tl.float32)

        # Materialize Q once and reshape to (BM, D_HEAD) for cube consumption.
        q_tile = tl.load(q_blk)
        q_tile = tl.reshape(q_tile, (BM, D_HEAD))

        n_kv_blocks = SEQ_LEN // BN

        # Cube scope
        # For each KV block:
        #   1. compute logits = q_tile · k_tile^T,  fixpipe -> logits_ub
        #   2. notify vector (event 0)
        #   3. wait vector to finish writing P into prob_l1_nz (event 1)
        #   4. compute partial O = P · v_tile,  fixpipe -> attn_value_ub
        #   5. notify vector (event 2)
        with al.scope(core_mode="cube"):
            for kv_step in range(0, n_kv_blocks, 1):
                _cube_compute_logits(q_tile, k_blk, logits_ub, D_HEAD, BN)
                al.sync_block_set ("cube",   "vector", 0, al.PIPE.PIPE_FIX,  al.PIPE.PIPE_V)
                al.sync_block_wait("vector", "cube",   1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
                _cube_apply_attn_to_value(prob_l1_nz, attn_value_ub, v_blk, D_HEAD, BM, BN)
                al.sync_block_set ("cube",   "vector", 2, al.PIPE.PIPE_FIX,  al.PIPE.PIPE_V)
                k_blk = tl.advance(k_blk, (BN, 0))
                v_blk = tl.advance(v_blk, (BN, 0))

        # Vector scope
        # For each KV block:
        #   1. wait logits in logits_ub  (event 0)
        #   2. online softmax: row_max, row_denom, rescale, prob
        #   3. write prob to prob_l1_nz in NZ layout
        #   4. notify cube (event 1)
        #   5. wait partial O in attn_value_ub (event 2)
        #   6. accumulate: out_acc = out_acc * rescale + partial
        with al.scope(core_mode="vector"):
            for kv_step in range(0, n_kv_blocks, 1):
                al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
                sub_vec_id = al.sub_vec_id()
                s_tile = bl.to_tensor(logits_ub)

                # ---- online softmax ----
                s_tile = s_tile * softmax_scale
                row_max_new = tl.maximum(
                    row_max,
                    tl.max(s_tile, 1, propagate_nan=True),
                    propagate_nan=tl.PropagateNan.ALL,
                )
                s_tile = s_tile - row_max_new[:, None]
                prob = tl.math.exp(s_tile)
                prob_cast = prob.to(elem_ty)

                # Re-shape P to NZ pre-form: (BN/16, HALF_M, 16)
                prob_nz_pre = tl.permute(
                    prob_cast.reshape(HALF_M, BN // 16, 16),
                    [1, 0, 2],
                )

                row_denom_local = tl.sum(
                    prob.reshape(HALF_M, BN), 1,
                )
                rescale = tl.math.exp(row_max - row_max_new)
                row_denom = row_denom * rescale + row_denom_local
                row_max = row_max_new

                # Copy P (NZ format) from UB to L1, into the half-slice
                # owned by this sub-vector.
                p_l1_slice = bl.subview(
                    prob_l1_nz,
                    [0, sub_vec_id * (HALF_M // 16), 0, 0],
                    [BN // 16, HALF_M // 16, 16, 16],
                    [1, 1, 1, 1],
                )
                prob_nz = prob_nz_pre.reshape(BN // 16, BM // 32, 16, 16)
                al.copy(
                    bl.to_buffer(prob_nz, al.ascend_address_space.UB),
                    p_l1_slice,
                )
                al.sync_block_set ("vector", "cube",   1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)

                al.sync_block_wait("cube",   "vector", 2, al.PIPE.PIPE_FIX,  al.PIPE.PIPE_V)
                out_acc = _vector_accumulate_partial(attn_value_ub, rescale, out_acc, D_HEAD, BM)

            out_acc = out_acc / row_denom[:, None]
            sub_vec_id = al.sub_vec_id()
            # out_blk_sub: tl.make_block_ptr, order=(1,0)
            #   base = out_ptr + bh_base_off + D_HEAD * sub_vec_id * HALF_M  ← sub_vec 行偏移
            #   shape=(SEQ_LEN, D_HEAD), strides=(s_o_m, s_o_d),
            #   offsets=(mb_idx * BM, 0), block_shape=(HALF_M, D_HEAD)
            tl.store(out_blk_sub, out_acc.to(out_ptr.type.element_ty))
```

**编译选项**：

```python
flash_attn_fwd_serial_kernel[grid](
    ...,
    debug=True,
    disable_auto_inject_block_sync=True,
    vf_merge_level=1,
)
```