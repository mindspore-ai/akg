---
name: triton-ascend-reduce
description: "适用于归约(reduce)类算子和含归约子步骤的复合算子（如归一化）的优化指南。典型算子包括：sum, mean, max, min, prod, argmax, argmin, cumsum, cumprod, softmax, logsoftmax, layernorm, rmsnorm, groupnorm, instancenorm, batchnorm, l1norm, l2norm, frobeniusnorm, var, std, average_pooling, sum_pooling 等。特别重要：当归约维度不是最后一维（如 dim=1 归约 shape=[B,F,D1,D2]），需要正确处理多维索引和两阶段归约。包含 PyTorch normalized_shape 多轴归一化语义说明。不适用于纯逐元素运算或矩阵乘法。如果算子是损失函数（先逐元素计算再全局归约），应选择 elementwise-reduce-fused 指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduce"
---

# Reduce 算子优化

> 适用于需要聚合多个值的归约操作

## 适用算子

**基础归约**: sum, mean, max, min, prod
**归一化**: softmax, logsoftmax, layernorm, rmsnorm, groupnorm, batchnorm
**统计**: variance, std

## 通用归约策略

### 1. 块内归约 + 原子操作

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    block_sum = tl.sum(data, axis=0)
    tl.atomic_add(output_ptr, block_sum)
```

## 非最后维度归约（关键难点）

当归约维度不是 tensor 的最后一维（例如对 shape `[B, F, D1, D2]` 沿 `dim=1` 归约），**不要使用 permute + reshape 预处理**，这会在 host 端产生巨大开销。正确做法是**在 kernel 内通过多维索引直接处理**。

### 核心思路

以 RMSNorm 对 `[B, F, D1, D2]` 沿 `dim=1`（F 维度）归约为例：
- **grid 第 0 维**：遍历 batch（B）
- **grid 第 1 维**：将 D1×D2 展平后分块，每个 program 处理一个 D1D2 块
- **kernel 内循环**：遍历归约维度 F 的分块
- **两阶段**：第一阶段累积统计量（如平方和），第二阶段用统计量归一化输出

### 标准模式：两阶段多维归约

```python
@triton.jit
def norm_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, F: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr, BLOCK_SIZE_D1D2: tl.constexpr,
):
    pid_b = tl.program_id(0)          # batch 维度
    pid_d1d2 = tl.program_id(1)       # D1*D2 展平后的块索引

    total_d1d2 = D1 * D2
    d1d2_start = pid_d1d2 * BLOCK_SIZE_D1D2
    d1d2_offsets = d1d2_start + tl.arange(0, BLOCK_SIZE_D1D2)
    d1d2_mask = d1d2_offsets < total_d1d2

    # 阶段1：沿 F 维度累积统计量
    accum = tl.zeros((BLOCK_SIZE_D1D2,), dtype=tl.float32)
    num_blocks_f = tl.cdiv(F, BLOCK_SIZE_F)

    for f_block in range(num_blocks_f):
        f_offsets = f_block * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        # 多维索引：x[b, f, d1, d2] → x_ptr + b*F*D1*D2 + f*D1*D2 + d1d2
        x_offsets = pid_b * F * total_d1d2 + f_offsets[:, None] * total_d1d2 + d1d2_offsets[None, :]
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        accum += tl.sum(x_tile * x_tile, axis=0)  # 以 RMS 为例

    rms = tl.sqrt(accum / F + eps)

    # 阶段2：归一化输出（同样的循环结构）
    for f_block in range(num_blocks_f):
        f_offsets = f_block * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        x_offsets = pid_b * F * total_d1d2 + f_offsets[:, None] * total_d1d2 + d1d2_offsets[None, :]
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        y_tile = x_tile / rms[None, :]
        tl.store(y_ptr + x_offsets, y_tile, mask=load_mask)
```

### Host 端启动

```python
def norm_forward(x, eps=1e-5):
    B, F, D1, D2 = x.shape
    y = torch.empty_like(x)
    total_d1d2 = D1 * D2
    BLOCK_SIZE_F = 16
    BLOCK_SIZE_D1D2 = 256
    grid = (B, triton.cdiv(total_d1d2, BLOCK_SIZE_D1D2))
    norm_kernel[grid](x, y, B, F, D1, D2, eps, BLOCK_SIZE_F, BLOCK_SIZE_D1D2)
    return y
```

### 关键要点

1. **不要 permute/reshape**：在 host 端做 `permute → contiguous → view(N, D)` 对大 tensor 开销极大
2. **多维索引公式**：`x[b, f, d1, d2]` 在连续内存中的偏移 = `b*F*D1*D2 + f*D1*D2 + d1*D2 + d2`，如果将 D1D2 展平则简化为 `b*F*total_d1d2 + f*total_d1d2 + d1d2`
3. **2D tile 加载**：用 `[:, None]` 和 `[None, :]` 构造 2D 偏移矩阵，一次 load 获取 `[BLOCK_F, BLOCK_D1D2]` 的数据
4. **归约沿 axis=0**：`tl.sum(tile, axis=0)` 沿 F 维度归约，保留 D1D2 维度的独立统计量
5. **grid 规模**：第二维为 `cdiv(D1*D2, BLOCK_SIZE_D1D2)`，对于较大的 D1*D2 可能超过 65535，需要注意 Ascend 的 grid 上限限制

## PyTorch 归一化/归约算子语义（重要）

### normalized_shape 多轴语义

`nn.LayerNorm(normalized_shape)` 中 `normalized_shape` 是 tuple 时，归一化范围是 **最后 `len(normalized_shape)` 个维度的乘积**，不是单个维度。

```python
# 示例：input shape = (B, F, D1, D2), normalized_shape = (F, D1, D2)
# 正确：归一化 F×D1×D2 = 最后 3 个维度，N = F * D1 * D2
# 错误：只归一化 F 维度

# kernel 中正确实现：
total_norm_size = F * D1 * D2  # normalized_shape 各维度的乘积
# 沿最后 len(normalized_shape) 个维度做 mean/var
```

### 损失函数归约

- `nn.MSELoss(reduction='mean')`: 对所有元素取均值
- `nn.CrossEntropyLoss`: 输入 logits `(N, C)` + targets `(N,)`, 内部含 log_softmax + nll_loss
- loss 函数多数是 **elementwise 计算 + 全局 reduce**，先按 elementwise 展平处理，最后用 `tl.sum` 或 `tl.atomic_add` 汇总