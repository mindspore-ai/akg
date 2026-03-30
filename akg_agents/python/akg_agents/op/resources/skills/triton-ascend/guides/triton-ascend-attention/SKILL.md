---
name: triton-ascend-attention
description: "适用于注意力(attention)机制类算子的优化指南。当算子的核心计算是 Transformer 风格的注意力运算时应选择此指南，典型算子包括：self_attention, cross_attention, multi_head_attention, flash_attention, scaled_dot_product_attention, causal_attention, masked_attention 等。涵盖 QKV 矩阵乘分块、在线 softmax、因果 mask 处理、Flash Attention 分块策略等关键技巧。不适用于不含注意力结构的普通矩阵乘法或归约运算。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "attention"
---

# Attention 算子优化

## 标准 Attention 计算流程

标准的 Scaled Dot-Product Attention:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 三个阶段

1. **QK^T 计算**: `scores = Q @ K^T / sqrt(d_k)`，计算注意力分数
2. **Softmax 归一化**: `attn_weights = softmax(scores)`，确保权重和为1
3. **加权求和**: `output = attn_weights @ V`，得到最终输出

### 标准实现的问题

```python
# 朴素实现（内存开销大）
scores = (Q @ K.T) / sqrt(d_k)  # (seq_len, seq_len)
attn_weights = softmax(scores)   # 需要存储完整注意力矩阵
output = attn_weights @ V
```

**问题**: 
- 需要存储 `(seq_len, seq_len)` 的注意力矩阵
- 内存占用: O(seq_len²)
- 对于长序列（seq_len = 4096），内存占用巨大

## Flash Attention 优化策略

Flash Attention 通过分块计算和在线 Softmax 避免存储完整注意力矩阵。

### 核心思想

1. **分块计算**: 将大矩阵分块处理，减少内存占用
2. **在线 Softmax**: 使用增量式 softmax 算法，分块计算，维护全局最大值和归一化因子
3. **避免存储**: 不存储完整注意力矩阵

### 在线 Softmax 算法

关键是维护全局统计量，逐块更新：

```python
# 初始化全局统计量
m_i = -float("inf")  # 全局最大值
l_i = 0.0           # 全局 exp 和
acc = 0.0           # 输出累加器

# 分块处理
for start_n in range(0, seq_len, BLOCK_SIZE):
    # 1. 加载当前块的分数
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))
    
    # 2. 更新全局最大值
    m_ij = tl.maximum(m_i, tl.max(scores, 0))
    
    # 3. 计算当前块的 exp 值（数值稳定化）
    scores = scores - m_ij
    p = tl.math.exp2(scores * 1.44269504)  # log2(e)
    
    # 4. 更新全局 exp 和
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
    l_i = l_i * alpha + l_ij
    
    # 5. 更新输出累加器
    acc = acc * alpha + p
    
    # 6. 更新全局最大值
    m_i = m_ij

# 最终归一化
acc = acc / l_i
```