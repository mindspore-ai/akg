---
name: triton-ascend-performance-improvement
description: |
  Triton Ascend 性能优化实战经验。从批量自适应搜索中提炼的通用优化模式，覆盖 tile 调优方法论、内存加载优化、reduction 优化、隐式广播、多 Pass 合并、数据访问重构等。
category: improvement
version: "1.0.0"
metadata:
  case_type: improvement
  backend: ascend
  dsl: triton_ascend
---

## Tile 尺寸选择方法

tile 尺寸需满足硬件存储约束（具体容量参考传入的硬件信息文档）：

**CUBE 路径 (matmul / tl.dot)**：tile 必须能放入 L0A/L0B/L0C
- 计算公式：`BLOCK_M × BLOCK_K × sizeof(dtype) ≤ L0A 容量`
- fp32 占用是 fp16 的 2 倍，需相应缩小 tile
- K 维度按 512B 对齐可提升带宽利用率

**VEC 路径 (elementwise / reduce)**：所有活跃 tensor 需放入 UB
- 计算公式：`BLOCK_SIZE × sizeof(dtype) × 活跃tensor数 × multi_buffer系数 ≤ UB 容量`
- 编译器 auto-multi-buffer 会将占用增至 2~3 倍
- kernel 中间变量（如 `tl.where` 产生的临时缓冲）也占用 UB

**调优策略**：从较大 tile 开始尝试，遇到 `ub overflow` / `cbuf overflow` 编译错误时逐级缩小。配合 `@triton.autotune` 自动选优。

## 内存加载优化

在 `tl.load` 时直接应用 mask 和填充值，而非先无条件加载再用 `tl.where` 筛选：

```python
# 次优：先加载再 where（多一次中间操作，可能触发 vsel 编译错误）
tile = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
tile = tl.where(mask, tl.load(ptr + offsets), 0.0)

# 推荐：直接在 load 时应用 mask
tile = tl.load(ptr + offsets, mask=mask, other=0.0)
```

## 隐式广播替代显式展开

需要将低维 tensor 广播到高维时，用维度扩展（`[:, None]`）利用 Triton 的隐式广播，避免 `tl.broadcast_to` 创建临时矩阵：

```python
# 次优：显式展开为完整矩阵
a_broadcast = tl.broadcast_to(a_tile[:, None], (BLOCK_M, BLOCK_N))
c_tile = a_broadcast * b_tile

# 推荐：隐式广播
c_tile = a_tile[:, None] * b_tile
```

## Reduction 最佳实践

1. **标量累加器**：每个核心用 `core_sum = 0.0` 局部累加，避免 tensor 索引问题
2. **单次原子写入**：循环结束后 `tl.atomic_add(out_ptr, core_sum)` 一次写入
3. **避免 host 端 permute**：非最后维度 reduce 直接在 kernel 内用多维索引

## Host 端预计算 stride

在 host 端预先计算 tensor 的 stride 并作为参数传入 kernel，而非在 kernel 内部计算，可以减少传参错误并辅助编译器优化：

```python
# host 端
stride_am, stride_ak = A.stride(0), A.stride(1)
kernel[grid](A_ptr, ..., stride_am, stride_ak, ...)
```

## 多 Pass 合并为单 Pass

当算子对同一数据进行多次独立遍历时（如 softmax 的 max→exp_sum→normalize，或 topk 的多次扫描），应合并为单次遍历。每减少一次 HBM 读取，理论上可获得等比例加速。多 pass 合并对归约维度较小（能放入单个 BLOCK）的场景效果最佳。

## Strided 访问模式重构

当算子涉及非连续的 strided 访问（如需要访问相邻元素的配对计算），应重构数据加载策略：
- 改为按语义分组处理，使相关元素落在同一 block 内
- 避免 `flat_idx % D` + 跨 stride 的随机访问
- 连续加载后在块内处理配对关系，利用数据局部性

Ascend 硬件对非连续访问有显著性能惩罚，重构后性能差距可达数倍至数十倍。

## Normalization 两阶段优于单 Pass

对于 LayerNorm/RMSNorm 类算子，两阶段方案（第一遍统计 mean/var，第二遍归一化）通常优于单 pass（在线统计）。
原因：单 pass 循环体内活跃 tensor 增多，UB 压力增大，编译器流水线优化效率下降。实测中 2-pass 比单 pass 快约 20%。
