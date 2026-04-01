---
name: triton-ascend-ascend-hardware-constraints
description: "Ascend 硬件约束与编译器限制速查。涵盖 CUBE/VEC 存储层级预算、bishengIR 编译器已知限制、strided access 性能特征。适用于所有 Triton Ascend 算子生成和调试场景。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "all"
---

# Ascend 硬件约束与编译器限制

## 1. 存储层级预算

### CUBE 路径（matmul / tl.dot）

Matmul 数据走 L0A/L0B/L0C，**不经过 UB**：

| 缓冲区 | 容量 | 用途 | 约束 |
|--------|------|------|------|
| L0A | 64 KB | 左矩阵 A tile (m0 × k0) | `m0 × k0 × sizeof(A.dtype) ≤ **KB` |
| L0B | 64 KB | 右矩阵 B tile (k0 × n0) | `k0 × n0 × sizeof(B.dtype) ≤ **KB` |
| L0C | 128 KB | 结果 C tile (m0 × n0)，支持累加 | `m0 × n0 × sizeof(C.dtype) ≤ **KB` |

以 fp16 为例（2 字节/元素），L0A 可容纳 32K 个元素，即 BLOCK_M=128, BLOCK_K=256 恰好 64KB。

以 fp32 为例（4 字节/元素），L0A 可容纳 16K 个元素，即 BLOCK_M=128, BLOCK_K=128 恰好 64KB。

### VEC 路径（element-wise / reduce / norm）

向量运算数据走 UB：

| 缓冲区 | 容量 | 说明 |
|--------|------|------|
| UB | 192 KB | 单 VEC 可用，编译器启用 auto-multi-buffer 后实际占用约 2-3 倍基础量 |

VEC 算子的 BLOCK_SIZE 需满足：所有活跃 tensor 的总大小 × multi-buffer 系数 ≤ 192KB。当 kernel 中有多个中间变量（如 tl.where 产生的临时缓冲）时，实际占用会显著高于 `BLOCK_SIZE × sizeof(dtype) × 输入数`。

## 2. bishengIR 编译器已知限制

### 2.1 range() 边界不可混用运行时变量

```python
# 编译器崩溃（bishengIR SIGABRT）
for k in range(start_n, start_m + BLOCK, BLOCK_K):
    ...
```

`start_n`、`start_m` 是运行时值，`BLOCK`、`BLOCK_K` 是 `tl.constexpr`。这种混合用法会导致编译器内部错误。

**规避方案**：使用全 constexpr 的 range，在循环体内用运行时 if 跳过无效迭代：

```python
for k in range(0, N, BLOCK_K):  # N 和 BLOCK_K 都是 constexpr
    # 可选：运行时条件跳过无效块
    ...
```

### 2.2 其他已知限制

- `while` 循环 → 用 `for` + `if` 替代
- `return` / `break` / `continue` → 用 mask 控制
- 复杂 `tl.where` 用于内存偏移 → 拆分为 if-else 静态分支
- BLOCK_SIZE 必须 < 65536

## 3. Strided memory access 的性能代价

Ascend 硬件对非连续内存访问有显著性能惩罚。当 kernel 的核心路径包含 stride > 1 的内存访问模式（如 pooling 的滑窗、dilated convolution 的间隔采样），Triton 生成的代码需要逐元素或小块 gather，而 CANN 原生算子可能使用硬件数据搬运单元（MTE）的专用模式，性能差距可达数十倍。

**建议**：
- 优先将数据在 host 侧重排为连续布局（如 `F.pad` + contiguous view），再用连续 load 处理
