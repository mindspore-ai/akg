---
name: tilelang-ascend-example-attention
description: "Sparse Flash Attention 的 TileLang Ascend 实现示例。展示 Cube+Vector 融合编程的完整 attention 模式：Cube 核 Q·K^T 和 P·V 两段 GEMM + workspace 跨核通信 + Vector 核 online softmax 累加 + T.Parallel 逐元素计算。当生成 attention 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "attention"
---

# Sparse Flash Attention — TileLang Ascend 实现示例

**编程模式**：Developer（Cube+Vector 自动融合 + workspace 跨核通信）

**关键技术点**：
- `workspace_idx=[4, 5, 6, 7, 8]` 声明 5 个 workspace 用于 Cube→Vector 跨核通信
- Cube 核：`T.gemm_v0` 两段矩阵乘（Q·K^T 和 P·V），结果写入 workspace
- Vector 核：online softmax 累加（`T.reduce_max` → `T.exp` → `T.reduce_sum` → rescale）
- `T.Parallel` 逐元素计算（softmax 减最大值、指数、缩放）
- `AUTO_CV_COMBINE: True + AUTO_CV_SYNC: True` 自动 Cube+Vector 融合与同步
- `T.alloc_L1` / `T.alloc_L0C` / `T.alloc_ub` 三级内存分配

```python
import tilelang
from tilelang import language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6, 7, 8], pass_configs=pass_configs)
def sparse_attention_fwd(
    heads, dim, tail_dim, topk, kv_stride,
    kv_group=1, sm_scale=None, is_causal=True, block_I=64,
):
    sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 if sm_scale is None else sm_scale

    batch = 1
    seq_len = 128
    seq_len_kv = 32768
    head_kv = heads // kv_group

    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]

    dtype = "float16"
    accum_dtype = "float"
    indices_dtype = "int32"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, "H padding requires kv_group=1"

    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64
    v_block = H_per_block // 2
    block_num = seq_len * REPLICATE_H * batch * kv_group

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        workspace_1: T.Tensor([block_num, BI, D], dtype),
        workspace_2: T.Tensor([block_num, BI, D_tail], dtype),
        workspace_3: T.Tensor([block_num, H_per_block, BI], accum_dtype),
        workspace_4: T.Tensor([block_num, H_per_block, BI], dtype),
        workspace_5: T.Tensor([block_num, H_per_block, D], accum_dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid % (seq_len * REPLICATE_H)
            by = cid // (seq_len * REPLICATE_H) % batch
            bz = cid // (seq_len * REPLICATE_H) // batch % kv_group

            q_l1 = T.alloc_L1([H_per_block, D], dtype)
            q_tail_l1 = T.alloc_L1([H_per_block, D_tail], dtype)
            kv_l1 = T.alloc_L1([BI, D], dtype)
            kv_tail_l1 = T.alloc_L1([BI, D_tail], dtype)
            acc_s_l1 = T.alloc_L1([H_per_block, BI], dtype)

            acc_s_l0c = T.alloc_L0C([H_per_block, BI], accum_dtype)
            acc_o_l0c = T.alloc_L0C([H_per_block, D], accum_dtype)

            acc_o = T.alloc_ub([v_block, D], accum_dtype)
            sumexp = T.alloc_ub([v_block], accum_dtype)
            m_i = T.alloc_ub([v_block], accum_dtype)
            indices_ub_ = T.alloc_ub([BI], indices_dtype)
            kv_ub = T.alloc_ub([D], dtype)
            kv_tail_ub = T.alloc_ub([D_tail], dtype)
            acc_s_ub = T.alloc_ub([v_block, BI], accum_dtype)
            m_i_prev = T.alloc_ub([v_block], accum_dtype)
            acc_s_ub_ = T.alloc_ub([v_block, BI], accum_dtype)
            sumexp_i_ub = T.alloc_ub([v_block], accum_dtype)
            acc_s_half = T.alloc_ub([v_block, BI], dtype)
            acc_o_ub = T.alloc_ub([v_block, D], accum_dtype)
            acc_o_half = T.alloc_ub([v_block, D], dtype)

            b_i = by
            g_i = bz
            s_i = bx // REPLICATE_H
            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], q_l1)
            T.copy(Q[b_i, s_i, H0:H1, D:], q_tail_l1)

            for _ in T.serial(NI):
                T.copy(workspace_1[cid, 0:BI, 0:D], kv_l1)
                T.copy(workspace_2[cid, 0:BI, 0:D_tail], kv_tail_l1)

                T.gemm_v0(q_l1, kv_l1, acc_s_l0c, transpose_B=True, init=True)
                T.gemm_v0(q_tail_l1, kv_tail_l1, acc_s_l0c, transpose_B=True)

                T.copy(acc_s_l0c, workspace_3[cid, 0:H_per_block, 0:BI])
                T.copy(workspace_4[cid, 0:H_per_block, 0:BI], acc_s_l1)

                T.gemm_v0(acc_s_l1, kv_l1, acc_o_l0c, init=True)

                T.copy(acc_o_l0c, workspace_5[cid, 0:H_per_block, 0:D])

            T.tile.fill(acc_o, 0.0)
            T.tile.fill(sumexp, 0.0)
            T.tile.fill(m_i, -(2.0**30))

            for i_i in range(NI):
                T.copy(Indices[b_i, s_i, g_i, i_i * BI : i_i * BI + BI], indices_ub_)

                for bi_i in range(BI // 2):
                    T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, :D], kv_ub)
                    T.copy(KV[b_i, indices_ub_[bi_i + vid * BI // 2], g_i, D:], kv_tail_ub)
                    T.copy(kv_ub, workspace_1[cid, bi_i + vid * BI // 2, :])
                    T.copy(kv_tail_ub, workspace_2[cid, bi_i + vid * BI // 2, :])

                T.tile.fill(acc_s_ub, 0.0)

                T.copy(m_i, m_i_prev)

                T.copy(workspace_3[cid, vid * v_block : vid * v_block + v_block, :], acc_s_ub_)

                for i, j in T.Parallel(v_block, BI):
                    acc_s_ub[i, j] = acc_s_ub[i, j] + acc_s_ub_[i, j]

                for i, j in T.Parallel(v_block, BI):
                    acc_s_ub[i, j] = acc_s_ub[i, j] * sm_scale

                T.reduce_max(acc_s_ub, m_i, dim=-1)

                for i in T.Parallel(v_block):
                    m_i[i] = T.max(m_i[i], m_i_prev[i])

                for i in T.Parallel(v_block):
                    m_i_prev[i] = m_i_prev[i] - m_i[i]

                for i in T.Parallel(v_block):
                    m_i_prev[i] = T.exp(m_i_prev[i])

                for h_i, j in T.Parallel(v_block, BI):
                    acc_s_ub[h_i, j] = acc_s_ub[h_i, j] - m_i[h_i]

                for i, j in T.Parallel(v_block, BI):
                    acc_s_ub[i, j] = T.exp(acc_s_ub[i, j])

                T.reduce_sum(acc_s_ub, sumexp_i_ub, dim=-1)

                for i in T.Parallel(v_block):
                    sumexp[i] *= m_i_prev[i]

                for i in T.Parallel(v_block):
                    sumexp[i] += sumexp_i_ub[i]

                for h_i, j in T.Parallel(v_block, D):
                    acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]

                T.copy(acc_s_ub, acc_s_half)
                T.copy(acc_s_half, workspace_4[cid, vid * v_block : vid * v_block + v_block, :])

                T.copy(workspace_5[cid, vid * v_block : vid * v_block + v_block, :], acc_o_ub)

                for i, j in T.Parallel(v_block, D):
                    acc_o[i, j] += acc_o_ub[i, j]

            for h_i, j in T.Parallel(v_block, D):
                acc_o[h_i, j] = acc_o[h_i, j] / sumexp[h_i]

            T.copy(acc_o, acc_o_half)
            T.copy(acc_o_half, Output[b_i, s_i, H0 + vid * v_block : H0 + v_block + vid * v_block, :])

    return main
```

**attention 类算子通用模式**：
1. **Cube 阶段**：Q·K^T 计算注意力分数 + P·V 计算加权值，结果通过 workspace 传递给 Vector
2. **Vector 阶段**：online softmax 累加（维护 row_max / row_denom / out_acc 三个 running state）
3. **workspace 通信**：Cube 写 workspace → Vector 读 workspace，实现跨核数据传递
4. **online softmax**：每步更新 `m_i = max(m_i_prev, m_i_new)`，`sumexp = sumexp * exp(m_i_prev - m_i) + sumexp_new`，`out = out * exp(m_i_prev - m_i) + o_partial`
