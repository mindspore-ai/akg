---
name: tilelang-ascend-memory
description: "Ascend NPU 内存访问优化策略，包括 UB（统一缓冲区）利用、数据布局优化、L1/L0 层级管理和数据搬运技巧。适用于内存带宽受限、需要优化数据搬运效率、或处理大规模数据的内核代码性能优化场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 内存访问优化

## Ascend NPU 内存层级

```
GM（全局内存）
  ↕ T.copy
L1（Cube 核缓存）/ UB（Vector 核缓冲）
  ↕ T.copy
L0A / L0B（矩阵输入寄存器）→ L0C（矩阵输出寄存器）
```

**不可跨级访问**，必须通过 T.copy 逐级搬运。

## 块大小选择原理

- **VEC 类算子**（element-wise、reduce 等）：数据需放入 UB（192KB/VEC）
  - `block_M * block_N * sizeof(dtype) * VEC_NUM` 需小于 UB 可用容量
- **CUBE 类算子**（matmul 等）：
  - 左矩阵放 L0A，右矩阵放 L0B，结果放 L0C
  - 具体容量参考硬件规格
- 数据传输按 **256 Bytes 对齐**

## 连续内存访问

非连续张量先 `.contiguous()` 转换：

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        x = x.contiguous()
        return self.kernel(x)
```

## 数据搬运优化

```python
# 标准搬运模式
T.copy(X[bx * block_M + vid * block_M // VEC_NUM, by * block_N], x_ub)

# 注意索引计算：vid 用于在 VEC_NUM 个 vector 核间分配工作
# block_M // VEC_NUM 是每个 vector 核处理的行数
```

## VEC_NUM 用法

```python
VEC_NUM = 2  # vector 核数量

with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    # vid 取值 0 或 1
    # 每个 vid 处理 block_M // VEC_NUM 行
    x_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
```

## 要点
- 优先 `.contiguous()` + 直接索引
- 合理选择 block 大小，避免 UB 溢出
- 注意 256B 对齐要求
