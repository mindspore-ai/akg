---
name: triton-cuda-attention
description: "Attention 算子的 Triton-CUDA 实现指南。包含经过验证的 Flash Attention 完整示例、各变体（Causal/GQA/MQA/RoPE）的差异改法、在线 Softmax 算法和常见错误"
category: implementation
version: "2.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
  operator_patterns: "attention"
  algorithms: "flash-attention, causal-attention, grouped-query-attention, multi-query-attention"
---

# Triton-CUDA Attention

## 在线 Softmax 核心算法

Flash Attention 用分块 + 在线 Softmax 将内存从 O(L²) 降到 O(L)。每处理一个 KV 块：

```python
# 维护三个状态: m_i(行最大值), l_i(exp和), acc(输出累加器)
qk = tl.dot(q, k) * sm_scale_log2e       # Q @ K^T，预乘 log2(e) 以便用 exp2
m_ij = tl.maximum(m_i, tl.max(qk, 1))    # 更新最大值
p = tl.math.exp2(qk - m_ij[:, None])     # 数值稳定的 exp（CUDA 用 exp2 更快）
alpha = tl.math.exp2(m_i - m_ij)         # 修正因子
l_i = l_i * alpha + tl.sum(p, 1)         # 修正并更新分母
acc = acc * alpha[:, None]               # 修正之前的累加结果
acc = tl.dot(p.to(v.dtype), v, acc)      # 加上当前块贡献
m_i = m_ij
# 循环结束后: output = acc / l_i[:, None]
```

## 完整示例：标准 Flash Attention

输入 Q/K/V: `(B, H, L, D)`，经过 A100 验证。

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attn_fwd_kernel(
    Q, K, V, Out,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N_CTX,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    # grid = (cdiv(L, BLOCK_M), B * H)
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // NUM_HEADS
    off_h = pid_bh % NUM_HEADS

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    # K shape 声明为 (D, N_CTX)，tl.dot(q, k) 直接得到 Q@K^T 无需转置
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset, shape=(N_CTX, D), strides=(stride_qm, stride_qd),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset, shape=(N_CTX, D), strides=(stride_om, stride_od),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, D), order=(1, 0))
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset, shape=(D, N_CTX), strides=(stride_kd, stride_kn),
        offsets=(0, 0), block_shape=(D, BLOCK_N), order=(0, 1))
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset, shape=(N_CTX, D), strides=(stride_vn, stride_vd),
        offsets=(0, 0), block_shape=(BLOCK_N, D), order=(1, 0))

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    q = tl.load(Q_block_ptr)
    sm_scale_log2e = sm_scale * 1.44269504

    for start_n in range(0, N_CTX, BLOCK_N):
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        # 输入 layout: (B, H, L, D)
        #   B = batch_size, H = num_heads, L = seq_len, D = head_dim
        # 若外部 layout 不同（如 (B,L,H,D)），需先 transpose 再 contiguous
        B, H, L, D = query.shape
        query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        out = torch.empty_like(query)
        sm_scale = 1.0 / math.sqrt(D)
        BLOCK_M, BLOCK_N = 64, 64
        D_padded = triton.next_power_of_2(D)
        grid = (triton.cdiv(L, BLOCK_M), B * H)
        _flash_attn_fwd_kernel[grid](
            query, key, value, out, sm_scale,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            L, NUM_HEADS=H, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, D=D_padded,
        )
        return out
```

## 变体改法（基于标准 FA 的差异）

### Causal Attention

两处修改：

```python
# 1. 循环上界：只遍历到当前 Q 块位置（节省约一半计算）
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)
for start_n in range(0, hi, BLOCK_N):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # ... 加载 k, 计算 qk ...

    # 2. 在 qk 上叠加因果 mask
    causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = qk + tl.where(causal_mask, 0.0, float("-inf"))
    # ... 后续在线 softmax 不变 ...
```

### GQA（Grouped-Query Attention）

Q 有 H_q 个 head，K/V 有 H_kv 个 head（H_q 是 H_kv 的整数倍）。仅改 head 索引：

```python
# kernel 参数增加: Q_NUM_HEADS, KV_NUM_HEADS
off_h_q = pid_bh % Q_NUM_HEADS
# 关键：用整除 // 做分组映射，不能用 % （% 是交错映射，与 PyTorch 语义不符）
off_h_kv = off_h_q // (Q_NUM_HEADS // KV_NUM_HEADS)

q_offset = off_b * stride_qb + off_h_q * stride_qh   # Q 用 Q head
k_offset = off_b * stride_kb + off_h_kv * stride_kh   # K 用 KV head
v_offset = off_b * stride_vb + off_h_kv * stride_vh   # V 用 KV head
o_offset = off_b * stride_ob + off_h_q * stride_oh    # Out 用 Q head
# host 侧 grid = (cdiv(L, BLOCK_M), B * H_q)
```

### MQA（Multi-Query Attention）

GQA 的特例：H_kv=1。K/V 没有 head 维度：

```python
# kernel: K/V stride 去掉 stride_kh/stride_vh
k_offset = off_b * stride_kb        # 只有 batch 偏移
v_offset = off_b * stride_vb
# host 侧: key = key.squeeze(1)，传 3 个 stride 而非 4 个
```

## 变体速查

| 变体 | K/V 形状 | Kernel 改动 | Host 改动 |
|------|----------|-------------|-----------|
| Causal | 同 Q | + causal_mask + 循环上界 hi | 无 |
| GQA | (B,H_kv,L,D) | head 映射 `//` | grid 按 H_q |
| MQA | (B,1,L,D) | K/V 无 head stride | squeeze(1) |

变体可自由组合，例如 Causal+GQA 同时应用 head 映射和 causal mask。

## 常见错误

| 错误 | 修复 |
|------|------|
| GQA head 映射用 `%` 而非 `//` | `off_h_kv = off_h_q // (H_q // H_kv)` |
| Triton 中用 `tensor[:, :half]` slice | 用 `tl.arange` 显式偏移 |
| runtime 变量标了 `tl.constexpr` | 只有编译期常量标 constexpr |
| 忘记 `acc = acc * alpha[:, None]` | m_i 更新后必须修正之前的 acc |
| 忘记最终 `acc / l_i` | 循环后归一化 |
| D 未 pad 到 2 的幂 | `D_padded = triton.next_power_of_2(D)` |
| 输入未 `.contiguous()` | stride 计算依赖连续内存 |

## 性能要点

- CUDA 上用 `tl.math.exp2` + 预乘 `sm_scale * 1.44269504`，比 `tl.exp` 快
- K shape 声明为 `(D, N_CTX)`，`tl.dot(q, k)` 直接得 Q@K^T 无需转置
- 用 `tl.make_block_ptr` + `tl.advance`，比手动偏移更简洁安全
- 累加器必须 float32，`tl.dot(p.to(v.dtype), v, acc)` 做精度转换
- Autotune: BLOCK_M/N 取 64 或 128，num_warps=4~8，num_stages=3~4
