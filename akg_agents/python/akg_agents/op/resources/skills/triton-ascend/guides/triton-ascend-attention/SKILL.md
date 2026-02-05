---
name: triton-ascend-attention
description: "Attention 算子优化策略和 Flash Attention 实现"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "attention"
  algorithms: "self-attention, cross-attention, flash-attention, scaled-dot-product-attention"
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

### 实现细节

**全局统计量**:
- `m_i`: 全局最大值，用于数值稳定化
- `l_i`: 全局 exp 和，用于最终归一化
- `acc`: 输出累加器，存储加权和

**更新策略**:
1. 每处理一个新块，更新 `m_i` 为全局最大值
2. 计算修正因子 `alpha = exp(m_i_old - m_i_new)`
3. 用 `alpha` 修正之前的累加结果
4. 添加当前块的贡献

## 完整 Flash Attention Kernel 示例

```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Output_ptr,
    seq_len, d_k,
    stride_q, stride_k, stride_v, stride_o,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前处理的查询位置
    q_idx = tl.program_id(0)
    
    # 加载 Q 向量（当前查询）
    q_offsets = tl.arange(0, d_k)
    q = tl.load(Q_ptr + q_idx * stride_q + q_offsets)
    
    # 初始化全局统计量
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((d_k,), dtype=tl.float32)
    
    # 分块处理 K 和 V
    for start_k in range(0, seq_len, BLOCK_SIZE):
        k_offsets = start_k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < seq_len
        
        # 加载 K 块: (BLOCK_SIZE, d_k)
        K_block = tl.load(
            K_ptr + k_offsets[:, None] * stride_k + q_offsets[None, :],
            mask=k_mask[:, None],
            other=0.0
        )
        
        # 计算注意力分数: q @ K^T
        scores = tl.sum(q[None, :] * K_block, axis=1) / tl.sqrt(float(d_k))
        scores = tl.where(k_mask, scores, -float("inf"))
        
        # 更新全局最大值
        m_ij = tl.maximum(m_i, tl.max(scores))
        
        # 数值稳定化的 exp
        scores = scores - m_ij
        p = tl.math.exp2(scores * 1.44269504)
        
        # 更新全局统计量
        l_ij = tl.sum(p)
        alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
        l_i = l_i * alpha + l_ij
        
        # 加载 V 块
        V_block = tl.load(
            V_ptr + k_offsets[:, None] * stride_v + q_offsets[None, :],
            mask=k_mask[:, None],
            other=0.0
        )
        
        # 更新输出累加器
        acc = acc * alpha + tl.sum(p[:, None] * V_block, axis=0)
        
        # 更新全局最大值
        m_i = m_ij
    
    # 最终归一化
    output = acc / l_i
    
    # 存储结果
    tl.store(Output_ptr + q_idx * stride_o + q_offsets, output)
```

## 优化要点

### 1. 数值稳定性

- **减去最大值**: 每个块计算前减去全局最大值
- **使用 exp2**: `tl.math.exp2(x * 1.44269504)` 而非 `tl.exp(x)`
- **处理无效位置**: mask 为 False 时用 `-float("inf")`

### 2. 内存优化

- **分块处理**: 不存储完整注意力矩阵
- **流式计算**: 边计算边更新，减少中间存储
- **float32 累加**: 即使输入是 fp16，也用 float32 累加

### 3. 并行策略

- **逐查询并行**: 每个程序处理一个查询位置
- `grid = (seq_len,)`: 每个查询独立并行
- 使用 VEC 或 CUBE 核心数（根据序列长度）

## 完整示例：Flash Attention Forward

```python
import torch
import triton
import triton.language as tl

@triton.jit
def flash_attention_forward(
    Q, K, V, Out,
    seq_len, d_k,
    BLOCK_SIZE: tl.constexpr,
):
    q_idx = tl.program_id(0)
    
    # 加载查询
    q_offsets = tl.arange(0, d_k)
    q = tl.load(Q + q_idx * d_k + q_offsets)
    
    # 初始化
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((d_k,), dtype=tl.float32)
    
    # 分块处理
    for k_start in range(0, seq_len, BLOCK_SIZE):
        k_idx = k_start + tl.arange(0, BLOCK_SIZE)
        k_mask = k_idx < seq_len
        
        # 加载 K 和 V
        K_block = tl.load(K + k_idx[:, None] * d_k + q_offsets[None, :], 
                         mask=k_mask[:, None], other=0.0)
        V_block = tl.load(V + k_idx[:, None] * d_k + q_offsets[None, :],
                         mask=k_mask[:, None], other=0.0)
        
        # 计算分数
        scores = tl.sum(q[None, :] * K_block, axis=1) / tl.sqrt(float(d_k))
        scores = tl.where(k_mask, scores, -float("inf"))
        
        # 在线 Softmax
        m_ij = tl.maximum(m_i, tl.max(scores))
        p = tl.math.exp2((scores - m_ij) * 1.44269504)
        l_ij = tl.sum(p)
        alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
        
        l_i = l_i * alpha + l_ij
        acc = acc * alpha + tl.sum(p[:, None] * V_block, axis=0)
        m_i = m_ij
    
    # 归一化并存储
    output = acc / l_i
    tl.store(Out + q_idx * d_k + q_offsets, output)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        batch_size, seq_len, d_k = Q.shape
        Output = torch.empty_like(Q)
        
        BLOCK_SIZE = 64
        grid = (batch_size * seq_len,)
        
        flash_attention_forward[grid](
            Q.view(-1, d_k), K.view(-1, d_k), V.view(-1, d_k),
            Output.view(-1, d_k),
            seq_len, d_k,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return Output.view(batch_size, seq_len, d_k)
```

## 性能检查清单

- [ ] 是否使用了在线 Softmax 避免存储完整注意力矩阵？
- [ ] 是否正确维护了全局最大值 m_i 和 exp 和 l_i？
- [ ] 是否使用修正因子 alpha 更新之前的累加结果？
- [ ] 是否使用 float32 进行累加（即使输入是 fp16）？
- [ ] 无效位置是否设置为 -inf？

## 常见错误

1. **忘记修正之前的结果**: 更新 m_i 后忘记用 alpha 修正 acc 和 l_i
2. **数值不稳定**: 没有减去最大值或使用 float32 累加
3. **边界处理错误**: 没有正确处理 mask 或使用 -inf
4. **内存开销大**: 存储了完整注意力矩阵
5. **归一化错误**: 最后忘记除以 l_i

## 扩展：Causal Attention

对于因果注意力（只能看到之前的位置），添加 mask：

```python
# 在计算 scores 后添加
causal_mask = k_idx <= q_idx
scores = tl.where(causal_mask & k_mask, scores, -float("inf"))
```
