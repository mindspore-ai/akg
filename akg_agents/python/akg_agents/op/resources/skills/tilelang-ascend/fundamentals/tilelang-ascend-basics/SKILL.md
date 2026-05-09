---
name: tilelang-ascend-basics
description: "TileLang Ascend 编程基础，包括核心概念（T.prim_func、T.Kernel、@tilelang.jit）、内核函数结构、Ascend NPU 专用约束和标准代码模式。适用使用 TileLang Ascend、需要了解基本语法结构的任意内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_patterns: "all"
---

# TileLang Ascend 编程基础

## 标准内核结构

```python
import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def relu(block_M, block_N, dtype="float16"):
    M = T.symbolic("M")
    N = T.symbolic("N")
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            x_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            y_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            T.copy(
                X[bx * block_M + vid * block_M // VEC_NUM, by * block_N],
                x_ub,
            )
            T.tile.relu(y_ub, x_ub)
            T.copy(
                y_ub,
                Y[bx * block_M + vid * block_M // VEC_NUM, by * block_N],
            )

    return main


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        block_M = 128
        block_N = 256
        func = relu(block_M, block_N, dtype="float16")
        self.kernel = tilelang.compile(
            func, out_idx=[-1], pass_configs=pass_configs, target="ascendc"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return self.kernel(x)
```

## 核心概念

### 1. @tilelang.jit

JIT 编译装饰器，将 TileLang DSL 编译为 NPU 可执行代码。

```python
@tilelang.jit(out_idx=[-1], pass_configs={...})
def my_kernel(...):
    @T.prim_func
    def main(...):
        ...
    return main
```

- `out_idx=[-1]`: 最后一个参数为输出，TileLang 自动分配输出张量
- `pass_configs`: 编译优化配置（见下方常用配置）

### 2. T.prim_func

定义 kernel 的主函数，声明输入输出张量。

```python
@T.prim_func
def main(A: T.Tensor((M, N), "float16"), B: T.Tensor((M, N), "float16")):
    ...
```

### 3. T.Kernel

定义并行执行上下文。**Ascend 只支持一维 block 数**。

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    # cid: core id
    # vid: vector id (当 VEC_NUM > 1 时)
```

**重要约束**：
- `is_npu=True` 必须指定
- 只接受一维 block 数（不支持 `T.Kernel(m, n, k)`）
- `threads` 参数只支持 1 或 2

### 4. 内存分配

| API | 层级 | 模式 |
|-----|------|------|
| `T.alloc_shared(shape, dtype)` | L1/UB（自动映射） | Developer |
| `T.alloc_fragment(shape, dtype)` | L0A/B/C（自动映射） | Developer |
| `T.alloc_ub(shape, dtype)` | UB（显式） | Expert |
| `T.alloc_L1(shape, dtype)` | L1（显式） | Expert |
| `T.alloc_L0A/L0B/L0C(shape, dtype)` | L0（显式） | Expert |
| `T.alloc_local(shape, dtype)` | 本地寄存器 | 通用 |

### 5. 数据搬运

```python
# GM -> UB
T.copy(X[bx * block_M, by * block_N], x_ub)

# UB -> GM
T.copy(y_ub, Y[bx * block_M, by * block_N])
```

## 常用 pass_configs

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步插入
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,  # 自动 CV 分离
}
```

## Ascend 关键约束

| 约束 | 说明 | 替代方案 |
|------|------|----------|
| **一维 Kernel** | `T.Kernel` 只接受一维 block 数 | 手动索引分解：cid // n_num, cid % n_num |
| **threads ≤ 2** | 不支持大值 threads | 默认不指定或设为 2 |
| **静态循环边界** | 循环次数不能依赖 tensor 值 | 预计算最大循环次数 + 条件判断 |
| **不支持 T.gemm** | GPU 通用版不可用 | 使用 `T.gemm_v0` 或 `T.mma` |

## 输出张量创建

- 输出张量用 `torch.empty` / `torch.empty_like`
- 输入必须先 `.contiguous()`

## 边界处理

```python
# 使用 T.min 处理边界
cur_k = T.min(block_size, K - ki * block_size)
T.copy(X[bi, ki * block_size], x_ub)
```
