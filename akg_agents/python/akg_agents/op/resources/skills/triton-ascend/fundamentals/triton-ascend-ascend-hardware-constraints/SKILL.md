---
name: triton-ascend-ascend-hardware-constraints
description: "Ascend 硬件约束与编译器限制速查。涵盖 CUBE/VEC 存储层级预算计算方法、bishengIR 编译器已知限制、strided access 性能特征。适用于所有 Triton Ascend 算子生成和调试场景。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "all"
---

# Ascend 硬件约束与编译器限制

> 不同型号的硬件具体容量不同，算子生成时会同步传入硬件信息文档，以下公式中的容量值请参考该文档。

## 1. 存储层级预算

### CUBE 路径（matmul / tl.dot）

Matmul 数据走 L0A/L0B/L0C，**不经过 UB**：

| 缓冲区 | 用途 | 约束公式 |
|--------|------|---------|
| L0A | 左矩阵 A tile (m0 × k0) | `m0 × k0 × sizeof(A.dtype) ≤ L0A容量` |
| L0B | 右矩阵 B tile (k0 × n0) | `k0 × n0 × sizeof(B.dtype) ≤ L0B容量` |
| L0C | 结果 C tile (m0 × n0)，支持累加 | `m0 × n0 × sizeof(C.dtype) ≤ L0C容量` |

**计算示例**（以某硬件 L0A = 64KB 为例）：
- fp16（2 字节/元素）：可容纳 32K 个元素 → BLOCK_M=128, BLOCK_K=256 恰好填满
- fp32（4 字节/元素）：可容纳 16K 个元素 → BLOCK_M=128, BLOCK_K=128 恰好填满

选择 tile 尺寸时，确保三个缓冲区都不超限。fp32 占用是 fp16 的 2 倍，需相应缩小 tile。

### VEC 路径（element-wise / reduce / norm）

向量运算数据走 UB：

| 缓冲区 | 用途 | 约束公式 |
|--------|------|---------|
| UB | 所有活跃 tensor 和中间变量 | `BLOCK_SIZE × sizeof(dtype) × 活跃tensor数 × multi_buffer系数 ≤ UB容量` |

编译器启用 `auto-multi-buffer` 后，实际占用约为基础量的 2~3 倍。kernel 中的中间变量（如 `tl.where` 产生的临时缓冲）也占用 UB，实际占用会显著高于 `BLOCK_SIZE × sizeof(dtype) × 输入数`。

**tile 选择策略**：从较大 BLOCK_SIZE 开始尝试，遇到 `ub overflow` 编译错误时逐级缩小。

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

### 2.2 复杂 mask + tl.where 导致 HiVM 错误

当嵌套 mask 组合传入 `tl.where` 时，编译器后端可能报 `hivm.hir.vsel: Unsupported op for finding the root alloc`。

**规避方案**：用乘法替代 tl.where，将 bool mask 转为 float 后与数据相乘：

```python
# 触发 hivm.hir.vsel 错误
a = tl.where(tri_mask & bounds_mask, a, 0.0)

# 规避：mask 转 float 后相乘
a = a * tri_mask.to(tl.float16) * bounds_mask.to(tl.float16)
```

### 2.3 其他编译器限制

详见 debugging 文档中的「禁止使用的语法」完整列表。

## 3. Strided memory access 的性能代价

Ascend 硬件对非连续内存访问有显著性能惩罚。当 kernel 的核心路径包含 stride > 1 的内存访问模式（如 pooling 的滑窗、dilated convolution 的间隔采样），Triton 生成的代码需要逐元素或小块 gather，而 CANN 原生算子可能使用硬件数据搬运单元（MTE）的专用模式，性能差距可达数十倍。

**建议**：
- 优先将数据在 host 侧重排为连续布局（如 `F.pad` + contiguous view），再用连续 load 处理
- matmul 的 K 维度按 512B 对齐可提升带宽利用率
