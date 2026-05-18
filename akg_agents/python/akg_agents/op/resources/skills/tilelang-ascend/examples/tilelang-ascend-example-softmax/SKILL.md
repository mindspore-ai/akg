---
name: tilelang-ascend-example-softmax
description: "Online Softmax 的 TileLang Ascend Developer 模式实现示例。展示 Vector 核编程：T.alloc_ub UB 分配、T.reduce_max/T.reduce_sum 归约、T.tile.* 向量指令、T.tile.broadcast 广播、低精度转高精度计算、pass_configs 自动同步。当生成 softmax / 归一化类算子时可参考此示例。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduction"
---

# Online Softmax — TileLang Ascend 实现示例（Developer 模式）

**编程模式**：Developer（自动同步 + 自动内存规划）

**关键技术点**：
- `T.alloc_ub` UB 显式分配
- `T.reduce_max` / `T.reduce_sum` 归约操作
- `T.tile.broadcast` 标量→向量广播
- `T.tile.cast` 低精度→高精度类型转换
- `pass_configs` 开启 `AUTO_SYNC` + `MEMORY_PLANNING`
- Online Softmax 算法：两遍扫描（max+sum → normalize）
- `VEC_NUM = 2` 双核并行，`sub_block_M = block_M // VEC_NUM`

```python
import tilelang
from tilelang import language as T

tilelang.cache.clear_cache()

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

CAST_MODE_LOW2HIGH = "CAST_NONE"
CAST_MODE_HIGH2LOW = "CAST_RINT"


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def online_softmax(M, N, block_M, block_N, dtype="float"):
    use_float32_compute = dtype in ["bfloat16", "float16"]
    cal_dtype = "float32" if use_float32_compute else dtype

    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)
    VEC_NUM = 2
    sub_block_M = block_M // VEC_NUM

    def cast_or_copy(dst, src, mode, count):
        if use_float32_compute:
            return T.tile.cast(dst, src, mode, count)
        else:
            return T.copy(src, dst)

    @T.prim_func
    def main(
        A: T.Tensor([M, N], dtype),
        B: T.Tensor([M, N], dtype),
    ):
        T.func_attr({"enable_auto_sync": True})
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            a = T.alloc_ub([sub_block_M, block_N], dtype)
            a_cal = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            tile_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            prev_max = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_max_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tile_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum = T.alloc_ub([sub_block_M, 1], cal_dtype)
            prev_sum_2d = T.alloc_ub([sub_block_M, block_N], cal_dtype)
            tmp_exp = T.alloc_ub([sub_block_M, 1], cal_dtype)

            T.tile.fill(prev_max, -T.infinity(cal_dtype))
            T.tile.fill(prev_sum, 0.0)

            for by in T.serial(n_num):
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N],
                    a,
                    pad_value=-T.infinity(cal_dtype),
                )
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.reduce_max(a_cal, tile_max, dim=-1)
                T.tile.max(tile_max, prev_max, tile_max)
                T.tile.sub(tmp_exp, prev_max, tile_max)
                T.tile.exp(tmp_exp, tmp_exp)
                T.tile.mul(tmp_exp, prev_sum, tmp_exp)
                T.tile.broadcast(tile_max_2d, tile_max)
                T.tile.sub(a_cal, a_cal, tile_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.reduce_sum(a_cal, tile_sum, dim=-1)
                T.tile.add(prev_sum, tile_sum, tmp_exp)
                T.copy(tile_max, prev_max)

            T.tile.broadcast(prev_max_2d, prev_max)
            T.tile.broadcast(prev_sum_2d, prev_sum)
            for by in T.serial(n_num):
                T.copy(
                    A[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N], a
                )
                cast_or_copy(a_cal, a, CAST_MODE_LOW2HIGH, sub_block_M * block_N)
                T.tile.sub(a_cal, a_cal, prev_max_2d)
                T.tile.exp(a_cal, a_cal)
                T.tile.div(a_cal, a_cal, prev_sum_2d)
                cast_or_copy(a, a_cal, CAST_MODE_HIGH2LOW, sub_block_M * block_N)
                T.copy(
                    a, B[bx * block_M + vid * sub_block_M : bx * block_M + (vid + 1) * sub_block_M, by * block_N : (by + 1) * block_N]
                )

    return main
```

**调用方式**：

```python
func = online_softmax(M, N, block_M, block_N, dtype="float16")
b = func(a)
```
