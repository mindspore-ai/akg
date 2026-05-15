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

## 关键性能优化：计算重组（延迟归约）

> **Ascend 上 `tl.sum`/`tl.max`/`tl.min` 等归约指令开销较大**，循环内每次迭代都调用归约会成为性能瓶颈。核心思路：**循环内只做逐元素累加（`+=`），循环结束后再执行一次归约**。

### 反模式 vs 正确范式

```python
# ❌ 反模式：循环内每次都调 tl.sum，产生 N/BLOCK_SIZE 次归约
total = 0.0
for offset in range(0, N, BLOCK_SIZE):
    block = tl.load(ptr + offset + tl.arange(0, BLOCK_SIZE), ...)
    total += tl.sum(block)  # 每次迭代都归约 → 开销大

# ✅ 正确：循环内只做逐元素累加，最后一次性归约
acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
for offset in range(0, N, BLOCK_SIZE):
    block = tl.load(ptr + offset + tl.arange(0, BLOCK_SIZE), ...)
    acc += block               # 逐元素 add，无归约开销
total = tl.sum(acc)            # 仅此一次归约
```

### 2D 场景（沿某一轴归约）

```python
# ❌ 反模式：循环内每次沿 axis=0 归约
acc_1d = tl.zeros((BLOCK_N,), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    tile = tl.load(...)  # [BLOCK_M, BLOCK_N]
    acc_1d += tl.sum(tile, axis=0)  # 每次迭代都归约

# ✅ 正确：保持 2D 累加器，最后一次性归约
acc_2d = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for m_start in range(0, M, BLOCK_M):
    tile = tl.load(...)  # [BLOCK_M, BLOCK_N]
    acc_2d += tile                      # 保持 2D，无归约
result = tl.sum(acc_2d, axis=0)         # 最后一次归约 → [BLOCK_N]
```

### 适用条件

- **可结合律操作**：sum（`+=`）、prod（`*=`）等满足结合律的操作均可使用此范式
- **非 sum 归约（max/min）也适用**：循环内用 `tl.maximum`/`tl.minimum` 逐元素取极值，最后一次 `tl.max`/`tl.min`
- **UB 容量权衡**：2D 累加器占用更多 UB（统一缓冲区），需确保 `BLOCK_M × BLOCK_N × dtype_size` 不超出 UB 容量。当 UB 不够时可适当减小 BLOCK_SIZE
- **掩码处理**：累加器初始化为归约的幺元（sum → 0、prod → 1、max → -inf、min → inf），用 `other=幺元` 处理边界

### 完整示例：Sum reduction over a dimension

```python
@triton.jit
def sum_reduce_kernel(
    x_ptr, y_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    NUM_CORES: tl.constexpr = 20,
):
    """Input X[B, M, N] → Output Y[B, N]，沿 M 轴求和"""
    pid = tl.program_id(0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_blocks = B * num_blocks_n

    for block_idx in range(pid, total_blocks, NUM_CORES):
        b_idx = block_idx // num_blocks_n
        n_start = (block_idx % num_blocks_n) * BLOCK_SIZE_N
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # 2D 累加器，延迟归约
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for m_start in range(0, M, BLOCK_SIZE_M):
            m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
            m_mask = m_offsets < M
            x_offset = b_idx * M * N + m_offsets[:, None] * N + n_offsets[None, :]
            x_block = tl.load(x_ptr + x_offset, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            acc += x_block  # 逐元素累加，不归约

        result = tl.sum(acc, axis=0)  # 循环结束后一次性归约
        tl.store(y_ptr + b_idx * N + n_offsets, result, mask=n_mask)
```

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

    # 阶段1：沿 F 维度累积统计量（计算重组：延迟归约）
    accum = tl.zeros((BLOCK_SIZE_F, BLOCK_SIZE_D1D2), dtype=tl.float32)
    num_blocks_f = tl.cdiv(F, BLOCK_SIZE_F)

    for f_block in range(num_blocks_f):
        f_offsets = f_block * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
        f_mask = f_offsets < F
        # 多维索引：x[b, f, d1, d2] → x_ptr + b*F*D1*D2 + f*D1*D2 + d1d2
        x_offsets = pid_b * F * total_d1d2 + f_offsets[:, None] * total_d1d2 + d1d2_offsets[None, :]
        load_mask = f_mask[:, None] & d1d2_mask[None, :]
        x_tile = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0)
        accum += x_tile * x_tile  # 保持 2D 累加，不在循环内归约

    rms = tl.sqrt(tl.sum(accum, axis=0) / F + eps)  # 循环结束后一次性归约

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
4. **计算重组**：循环内 `accum += x_tile * x_tile` 保持 2D 累加，循环结束后 `tl.sum(accum, axis=0)` 一次性归约，避免每次迭代调用 `tl.sum`
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