---
name: tilelang-ascend-debugging
description: "TileLang-Ascend 算子编码常见编码模式问题。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang-Ascend 算子编码常见编码模式问题

## Checklist

生成代码后逐项检查：

### 基础检查

| # | 检查项 |
|---|--------|
| 1 | `out_idx` 与函数签名中的输出参数位置一致 |
| 2 | `block_M // VEC_NUM` 在 buffer 分配和索引中一致使用 |
| 3 | 所有 `T.alloc_ub` 的 shape 乘积不超 UB 容量 |
| 4 | Expert 模式有 `T.Scope("V")` 和 `T.barrier_all()` |
| 5 | Developer 模式有对应的 `pass_configs` |
| 6 | 测试包含至少 2 个配置（小规模 + 典型规模） |
| 7 | golden 函数使用 PyTorch 标准实现 |

### 融合算子检查

| # | 检查项 | 说明 |
|---|--------|------|
| 8 | **workspace_idx 与函数签名一致** | workspace 参数位置正确 |
| 9 | **AUTO_CV_COMBINE / AUTO_CV_SYNC 配置** | Developer 模式需开启 |
| 10 | **Cube → workspace → Vector 数据流正确** | T.copy 搬运路径完整 |
| 11 | **核分离方式与 pass_configs 匹配** | Developer 模式无需显式 T.Scope |

## 1. 如何处理动态shape?

使用 `T.symbolic`：
```python
N = T.symbolic('N', 'int32')
```

## 2. 如何实现带参数的算子?

使用函数参数传递：
```python
def my_op(M, N, block_M, param1=0.1, dtype="float"):
    @T.prim_func
    def main(...):
        T.tile.add(a_ub, a_ub, param1)
```

## 3. 如何处理非2D数据?

调整索引和分块策略：
```python
@T.prim_func
def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype)):

@T.prim_func
def main(A: T.Tensor((B, M, N), dtype), ...):
```

## 4. 如何优化内存使用?

1. 开启自动内存规划
2. 复用中间buffer
3. 避免不必要的buffer分配

## 5. 标量与向量运算必须用 T.tile API，标量只能放在第二个操作数，部分 API 不支持标量

标量与向量之间的算术运算不能用 `+ - * /` 等运算符，必须用 `T.tile` 系列 API。

正确做法：

| 想写的表达式 | 实现方式 |
|------------|---------|
| `1.0 - x` | `T.tile.mul(x, x, -1.0)` 再 `T.tile.add(x, x, 1.0)` |
| `2.0 / x` | `T.tile.div(dst, T.broadcast(2.0, shape), x)`（**重要**：`T.tile.reciprocal`精度不足，禁止使用） |
| `x + 1.0` | `T.tile.add(x, x, 1.0)` |

## 6. T.Kernel(n_num) 和 T.serial(n_num) 不要混用

- `T.Kernel(n_num, is_npu=True) as (cid, vid)` 决定 launch 多少个 block 并行执行，每个 block 跑一遍 kernel body
- `for by in T.serial(n_num):` 是单个 block 内部的串行循环，用于一个核需要分多次处理多块数据

两者语义独立，不要在 `T.Kernel(n_num)` 的 body 里再用 `for by in T.serial(n_num)`：

```python
# 错误：n_num 同时控制 block 数又控制 serial 循环，语义重复
with T.Kernel(n_num, is_npu=True) as (cid, vid):
    for by in T.serial(n_num): ...  # 不需要这个循环，每个 (cid, vid) 直接处理自己的 partition
```

## 7. 禁止在 Kernel 内使用 Python 内置函数

TileLang kernel 中的 TVM `Expr` 对象是符号化表达式，不能使用 Python 的内置函数（如 `min()`、`max()`、`and`、`or`、`not`）进行操作。

```python
# 错误 ❌ - 禁止使用 Python min()
hw_end = min(hw_start + block_HW, H * W)       # ❌ 不支持 Python min

# 正确 ✅ - 使用T.min
hw_end = T.min(hw_start + block_HW, H * W)
```