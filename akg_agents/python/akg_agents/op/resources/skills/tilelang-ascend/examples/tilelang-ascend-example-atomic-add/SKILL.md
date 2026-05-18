---
name: tilelang-ascend-example-atomic-add
description: "atomic_add 的 TileLang Ascend Developer 模式实现示例。当需要生成all-reduce算子或多核并行累加到同一输出区域的 reduce 类算子时可参考此示例。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# atomic_add（多 block 原子累加）— TileLang Ascend 实现示例（Developer 模式）

**编程模式**：Developer（`pass_configs` 自动同步 + 内存规划）

**关键技术点**：
- `T.tile.atomic_add(dst_gm, src_local)` — 多 block/core 原子累加到同一 GM 区域
- 调用前必须清零 GM 输出（`torch.zero_()` 或 kernel 内 `T.tile.fill` + `T.copy` 清零）
- UB → GM 路径：Vector 核计算结果从 UB 原子写回 GM
- L0C → GM 路径：Cube 核 GEMM 结果从 L0C 原子写回 GM
- `pass_configs` 开启 `AUTO_SYNC` + `MEMORY_PLANNING`，混合模式下无需手写 `T.Scope("V")` 或 `T.barrier_all()`

## 场景说明

当多个 block/core 需要将各自的 partial result 累加到同一 GM 输出区域时，普通 `T.copy` 会互相覆盖，必须使用 `T.tile.atomic_add` 保证原子累加。

典型场景：
- **Split-K GEMM**：多个 block 沿 K 维切分，各自计算部分矩阵乘结果，原子累加到同一 GM 输出
- **行归约非整除**：多 block 对同一行不同区间做 reduce，原子累加行结果到 GM
- **全 reduce**：所有 block 对同一区域贡献部分值，原子累加汇总

## 示例一：UB → GM 原子累加（1D）

多个 block 各自 fill 1.0 到 UB，然后原子累加到同一 GM 输出。最终 GM 每个元素的值 = num_blocks × VEC_NUM。

```python
import tilelang
from tilelang import language as T
import torch

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

num_blocks = 4
tile_n = 32
dtype = "float32"

@tilelang.jit(pass_configs=pass_configs)
def atomic_add_1d(num_blocks, tile_n, dtype):
    @T.prim_func
    def main(C: T.Tensor((tile_n,), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_n,), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0], src_ub)

    return main
```

## 示例二：UB → GM 原子累加（2D region）

多个 block 各自 fill 1.0 到 2D UB，然后原子累加到同一 2D GM 区域。

```python
tile_m = 4
tile_n = 32
dtype = "float32"

@tilelang.jit(pass_configs=pass_configs)
def atomic_add_2d(num_blocks, tile_m, tile_n, dtype):
    @T.prim_func
    def main(C: T.Tensor((tile_m, tile_n), dtype)):
        with T.Kernel(num_blocks, is_npu=True) as (cid, vid):
            src_ub = T.alloc_ub((tile_m, tile_n), dtype)

            T.tile.fill(src_ub, 1.0)
            T.tile.atomic_add(C[0, 0], src_ub)

    return main
```