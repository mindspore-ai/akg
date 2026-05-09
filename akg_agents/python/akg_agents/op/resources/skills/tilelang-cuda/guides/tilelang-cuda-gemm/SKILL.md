---
name: tilelang-cuda-gemm
description: "TileLang CUDA GEMM 完整性能优化指南，涵盖基础模板、Swizzling、L2 Cache Rasterization、Auto-Pipelining、Persistent Kernel、Autotuning、Split-K、Stream-K、Fine-grained MMA 等高级优化技术。基于 tilelang 官方 examples/gemm/ 最佳实践。适用于所有矩阵乘法及其变体（BMM、FP8 GEMM、Int4 GEMM、Dequant GEMM 等）的内核代码生成和优化场景"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  operator_patterns: "matmul, bmm, fp8_gemm, int4_gemm, dequant_gemm"
structure:
  optimization_levels:
    level1: "basic_gemm"
    level2: "swizzling_rasterization"
    level3: "autotuning_persistent"
    level4: "splitk_streamk"
    level5: "fine_grained_mma"
---

# TileLang GEMM 性能优化完全指南

本 Skill 基于 tilelang 官方 `examples/gemm/` 和 `docs/programming_guides/instructions.md` 整理，覆盖从基础到高端的所有 GEMM 优化技术。

## 1. 基础 GEMM 模板

所有 GEMM 的核心模式如下：

```python
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm
```

### 关键概念

1. **内存层次**
   - `T.alloc_shared`: 共享内存 (__shared__)
   - `T.alloc_fragment`: 寄存器片段 (Tensor Core local)
   - 数据流: Global → Shared → Fragment(GEMM) → Shared → Global

2. **分块参数推荐值**
   - `block_M, block_N = 128, 128` (经典), `256, 256` (大矩阵), `128, 256` (非对称)
   - `block_K = 32` (推荐默认), `64` (SM90+)
   - `threads = 128` (GEMM 经典值), `256` (某些场景)
   - `num_stages = 3` (推荐), `2` (最小共享内存), `4` (SM90+)

3. **累积精度**
   - `accum_dtype=T.float32` 保证精度，尤其对 float16 输入
   - FP8 场景需要 `accum_dtype=T.float32` + 2x accumulate 模式

## 2. 高级优化：Swizzling + Rasterization + Parallel Copy

以下优化组合使用可显著提升 GEMM 性能：

```python
import tilelang.language as T
from tilelang.cuda.intrinsics import make_mma_swizzle_layout as make_swizzle_layout


@tilelang.jit(out_idx=[-1])
def matmul_optimized(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            # 优化1: Swizzle shared memory layout → 避免 bank conflict
            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })

            # 优化2: Rasterization → 提升 L2 cache 命中率
            T.use_swizzle(panel_size=10, enable=True)

            T.clear(C_local)
            for idx in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, idx * block_K], A_shared)

                # 优化3: Parallel copy B → 多线程并行加载
                for ko, j in T.Parallel(block_K, block_N):
                    B_shared[ko, j] = B[idx * block_K + ko, bx * block_N + j]

                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
```

### 优化点详解

| 优化技术 | 作用 | 何时使用 |
|----------|------|----------|
| `T.annotate_layout` | 共享内存 Swizzling，避免 bank conflict | 任何 GEMM |
| `T.use_swizzle` | 网格级 L2 cache 光栅化 | 大矩阵 M,N > 2048 |
| `T.Parallel(..., ...)` copy | 多线程并行全局→共享加载 | 当 `T.copy` 是瓶颈时 |
| 中间 `C_shared` 缓冲 | 避免 fragment→global 直接写入 | 需要 swizzle 布局时 |

### ⚠️ 注意事项

- `T.annotate_layout` 的 swizzle layout 需要 `from tilelang.cuda.intrinsics import make_mma_swizzle_layout`
- `T.use_swizzle(panel_size=10)` 的 panel_size 控制光栅化粒度，10 是推荐默认值
- 当使用 Parallel copy 时，`B_shared` 的索引必须包含 `bx`, `ko`, `j` 的完整映射

## 3. Autotuning

tilelang 提供 `@tilelang.autotune` 和 `AutoTuner` 进行自动调优：

### 3.1 手动搜索空间

```python
import tilelang as tl
import tilelang.language as T
import torch
from tilelang.autotuner import AutoTuner


def ref_program(A, B):
    return A @ B.T


def get_configs(M, N, K):
    """生成调优搜索空间"""
    import itertools
    block_M_list = [64, 128, 256]
    block_N_list = [64, 128, 256]
    block_K_list = [32, 64]
    num_stages = [0, 1, 2, 3]
    thread_num = [128, 256]
    enable_rasterization = [True, False]

    _configs = list(itertools.product(
        block_M_list, block_N_list, block_K_list,
        num_stages, thread_num, enable_rasterization,
    ))
    return [
        {
            "block_M": c[0], "block_N": c[1], "block_K": c[2],
            "num_stages": c[3], "thread_num": c[4],
            "enable_rasteration": c[5],
        }
        for c in _configs
    ]


def autotune_gemm(M, N, K):
    def kernel(block_M=None, block_N=None, block_K=None, num_stages=None,
               thread_num=None, enable_rasteration=None):
        dtype = T.bfloat16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                C_shared = T.alloc_shared((block_M, block_N), dtype)
                T.use_swizzle(panel_size=10, enable=enable_rasteration)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * block_M, bx * block_N])
        return main

    autotuner = AutoTuner.from_kernel(
        kernel=kernel, configs=get_configs(M, N, K)
    ).set_compile_args(
        out_idx=[-1], target="auto",
    ).set_profile_args(
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program, skip_check=False, backend="event",
    )
    return autotuner.run(warmup=3, rep=20)
```

### 3.2 使用 Roller (BitBLAS) 智能搜索

```python
from tilelang.carver.template import MatmulTemplate
from tilelang.carver.arch import CUDA


def get_roller_configs(M, N, K, topk=20):
    arch = CUDA("cuda")
    carve_template = MatmulTemplate(
        M=M, N=N, K=K,
        in_dtype=T.float16, out_dtype=T.float16, accum_dtype=T.float32,
    ).with_arch(arch)
    roller_hints = carve_template.recommend_hints(topk=topk)
    if roller_hints is None:
        raise ValueError("No Roller Hints Found for TensorCore Scheduling")
    configs = []
    for hint in roller_hints:
        block_m, block_n = hint.block
        warp_m, warp_n = hint.warp
        block_rows, block_cols = block_m // warp_m, block_n // warp_n
        configs.append({
            "block_M": block_m, "block_N": block_n,
            "block_K": hint.rstep[0],
            "num_stages": hint.pipeline_stage if hint.pipeline_stage > 1 else 0,
            "thread_num": block_rows * block_cols * 32,
            "enable_rasteration": hint.rasterization_plan is not None,
        })
    return configs
```

### 3.3 SM 版本的 Heuristic Config

```python
import torch


def get_heuristic_config():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.cuda.current_device()
    sm_major, _ = torch.cuda.get_device_capability(device)
    sm_version = sm_major * 10
    if sm_version == 80:  # A100
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 2, "thread_num": 128, "enable_rasteration": True}
    elif sm_version == 90:  # H100
        return {"block_M": 128, "block_N": 256, "block_K": 64,
                "num_stages": 3, "thread_num": 256, "enable_rasteration": True}
    else:  # 默认
        return {"block_M": 128, "block_N": 256, "block_K": 32,
                "num_stages": 0, "thread_num": 128, "enable_rasteration": True}
```

## 4. Persistent Kernel

Persistent Kernel 适用于大量输出 tile 的场景，通过将网格限制为 `sm_num` 个 block 来减少 kernel launch overhead：

```python
import tilelang.language as T
from tilelang.carver.arch import driver


@tilelang.jit(out_idx=[-1])
def matmul_persistent(M, N, K, block_M, block_N, block_K, threads, num_stages,
                      dtype=T.float16, accum_dtype=T.float32):
    sm_num = driver.get_num_sms()
    m_blocks = T.ceildiv(M, block_M)
    n_blocks = T.ceildiv(N, block_N)
    waves = T.ceildiv(m_blocks * n_blocks, sm_num)
    group_size = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(10)

            for w in T.serial(waves):
                tile_id = sm_num * w + block_id
                bx = (tile_id // group_size) % m_blocks
                by = (tile_id % group_size) + (tile_id // group_size) // m_blocks * group_size

                if bx * block_M < M and by * block_N < N:
                    T.clear(C_local)
                    for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                        T.copy(A[bx * block_M, k * block_K], A_shared)
                        T.copy(B[k * block_K, by * block_N], B_shared)
                        T.gemm(A_shared, B_shared, C_local)

                    T.copy(C_local, C_shared)
                    T.copy(C_shared, C[bx * block_M, by * block_N])

    return main
```

### Persistent Kernel vs T.Persistent 原语

tilelang 还支持 `T.Persistent` 原语（更简洁）：

```python
    @T.prim_func
    def main_persistent(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(sm_num, threads=threads) as (block_id):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            for bx, by in T.Persistent(
                [T.ceildiv(M, block_M), T.ceildiv(N, block_N)],
                sm_num, block_id
            ):
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[bx * block_M, k * block_K], A_shared)
                    T.copy(B[k * block_K, by * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[bx * block_M, by * block_N])

    return main_persistent  # 返回这个版本
```

### 何时使用 Persistent Kernel

| 场景 | 推荐 | 原因 |
|------|------|------|
| 大矩阵 (M,N > 4096) | ✅ 是 | 减少 kernel launch overhead，提升 SM 利用率 |
| 小矩阵 (M,N < 1024) | ❌ 否 | 额外开销不值得 |
| 多 CTA 场景 | ✅ 是 | 2-CTA persistent 对 SM100+ 效果更好 |

## 5. Split-K

Split-K 适用于 K 维度极大的场景（K/M > 4 时效果明显）：

```python
import tilelang.language as T


@tilelang.jit(out_idx=[2])
def matmul_splitk(M, N, K, block_M, block_N, block_K, split_k,
                  dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)

            T.clear(C_local) if bz == 0 else None

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K // split_k, block_K), num_stages=3):
                T.copy(A[bz * (K // split_k) + by * block_M, k * block_K], A_shared)
                T.copy(B[bz * (K // split_k) + k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main
```

### Split-K 要点

- **额外 grid 维度**: `split_k` 作为第 3 维
- **原子累加**: 使用 `T.atomic_add` 合并 partial results
- **适用场景**: K >> M, K >> N（如 K=16384, M=N=1024）
- **典型 split_k 值**: 2, 4, 8

## 6. Transpose B (B^T * A 模式)

当输入 B 的形状为 `(N, K)` 而非 `(K, N)` 时：

```python
@tilelang.jit(out_idx=[-1])
def matmul_transpose_b(M, N, K, block_M, block_N, block_K,
                       dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), dtype)

            T.use_swizzle(panel_size=10, enable=True)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main
```

**关键字**: `transpose_B=True` 自动处理 B 的转置计算。

## 7. FP8 GEMM

FP8 使用 MMA 指令而非自动 T.gemm dispatch，需要注意：

```python
import tilelang.language as T
from tilelang.utils import determine_fp8_type


@tilelang.jit(out_idx=[-1])
def matmul_fp8(M, N, K, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    @T.prim_func
    def gemm_fp8(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_fp8


# 调用
# dtype = determine_fp8_type()  # e4m3
# kernel = matmul_fp8(1024, 1024, 1024, 128, 128, 64, dtype)
# a = torch.randn(M, K).cuda().to(dtype)
# b = torch.randn(N, K).cuda().to(dtype)
# c = kernel(a, b)
```

### FP8 要点

- **使用 `transpose_B=True`**: FP8 GEMM 需要 B 的形状为 `(N, K)`
- **dtype**: `determine_fp8_type()` 返回 e4m3；`determine_fp8_type("e5m2")` 返回 e5m2
- **累积精度**: 始终使用 `accum_dtype=T.float32`
- **验证**: FP8 验证使用 `calc_diff` 而非 `torch.testing.assert_close`

## 8. Fine-grained MMA（细粒度 MMA）

当自动 `T.gemm` 不满足需求时（如 dequantize GEMM 需要自定义 layout），使用 `TensorCoreIntrinEmitter`：

```python
from tilelang.intrinsics import TensorCoreIntrinEmitter, make_mma_swizzle_layout


def dequant_gemm(M, N, K, in_dtype, out_dtype, accum_dtype):
    micro_size_x = micro_size_y = micro_size_k = 16
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 32
    warp_col_tiles = 32
    chunk = 32
    shared_scope = "shared.dyn"

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype, b_dtype=in_dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True,
        block_row_warps=block_row_warps, block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles, warp_col_tiles=warp_col_tiles, chunk=chunk,
    )

    @T.prim_func
    def main(A: T.Tensor((M, K), in_dtype),
             B: T.Tensor((N, K), in_dtype),
             C: T.Tensor((M, N), out_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=32 * block_row_warps * block_col_warps) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype, scope=shared_scope)
            C_shared = T.alloc_shared(
                (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y),
                out_dtype, scope=shared_scope)
            A_local = T.alloc_local((block_row_warps * (micro_size_x * micro_size_k) // 32), in_dtype)
            B_local = T.alloc_local((block_col_warps * (micro_size_y * micro_size_k) // 32), in_dtype)
            C_local = T.alloc_local((
                block_row_warps * block_col_warps * (micro_size_x * micro_size_y) // 32), accum_dtype)

            T.annotate_layout({
                A_shared: make_swizzle_layout(A_shared),
                B_shared: make_swizzle_layout(B_shared),
            })
            T.use_swizzle(panel_size=10)
            T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=2):
                # Parallel copy from global to shared
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                # Fine-grained MMA
                for ki in T.serial(block_K // micro_size_k):
                    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                    mma_emitter.mma(A_local, B_local, C_local)

            mma_emitter.stmatrix(C_local, C_shared)

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[
                    i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y]

    return main
```

### Fine-grained MMA 适用场景

- **Dequantize GEMM** (W4A8, FP4, Int4): 需要在 dequantize 后直接进行 MMA
- **自定义 layout**: 自动 `T.gemm` 的 layout 不满足需求
- **极致性能**: 需要完全控制 ldmatrix/mma/stmatrix 序列

## 9. Profiling / Benchmark

### 9.1 使用 Profiler 类

```python
kernel = matmul(4096, 4096, 4096, 128, 128, 32)
profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Randn)

# 正确性验证
profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)

# 评测
latency = profiler.do_bench(backend="event")    # CUDA Event (默认)
# latency = profiler.do_bench(backend="cupti")  # CUPTI profiler (更精确)
# latency = profiler.do_bench(backend="cudagraph")  # CUDA graph

# TFLOPs
M, N, K = 4096, 4096, 4096
tflops = 2 * M * N * K / latency * 1e-9
```

### 9.2 使用 do_bench 函数

```python
from tilelang.profiler import do_bench

kernel = matmul(4096, 4096, 4096, 128, 128, 32)
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)

latency = do_bench(
    lambda: kernel(a, b),
    warmup=25, rep=100, backend="event", return_mode="min"
)
```

### 9.3 Benchmark 参数推荐

| 参数 | 值 | 说明 |
|------|-----|------|
| `warmup` | 25ms | warmup 目标时间 |
| `rep` | 100ms | 评测目标时间 |
| `backend` | "event" | 默认，CUDA Event 计时 |
| `backend` | "cupti" | CUPTI profiler，更精确 |
| `backend` | "cudagraph" | CUDA graph replay |
| `return_mode` | "min" | 推荐，取最小值 |

## 10. 指令速查表

### 数据移动

| 指令 | 用途 | 示例 |
|------|------|------|
| `T.copy(src, dst)` | 同步拷贝 | `T.copy(A[...], A_shared)` |
| `T.async_copy(src, dst)` | 异步拷贝 (cp.async) | `T.async_copy(A[...], A_shared)` + `T.ptx_wait_group(0)` |
| `T.tma_copy(src, dst)` | TMA 异步拷贝 (SM90+) | `T.tma_copy(desc, A_shared)` |

### 内存分配

| 指令 | 用途 | 示例 |
|------|------|------|
| `T.alloc_shared` | 共享内存 | `T.alloc_shared((128, 32), "float16")` |
| `T.alloc_fragment` | 寄存器片段 | `T.alloc_fragment((128, 128), "float")` |
| `T.alloc_local` | 线程本地 | `T.alloc_local((1,), "float32")` |
| `T.alloc_tmem` | Tensor Memory (SM100) | `T.alloc_tmem((128, 256), "float32")` |

### 计算

| 指令 | 用途 | 示例 |
|------|------|------|
| `T.gemm(A, B, C)` | Tile GEMM | `T.gemm(A_shared, B_shared, C_local)` |
| `T.gemm(A, B, C, transpose_B=True)` | B 转置模式 | `T.gemm(A_s, B_s, C_l, transpose_B=True)` |
| `T.clear(buf)` | 清零 | `T.clear(C_local)` |
| `T.reduce_max/min/sum` | 归约 | `T.reduce_sum(input, output, dim=0)` |

### 循环控制

| 指令 | 用途 | 示例 |
|------|------|------|
| `T.Pipelined(n, stages)` | 软件流水线 | `for k in T.Pipelined(..., num_stages=3)` |
| `T.Parallel(d1, d2)` | 并行循环 | `for i, j in T.Parallel(128, 128)` |
| `T.serial(n)` | 串行循环 | `for ki in T.serial(block_K // 16)` |
| `T.Persistent(...)` | Persistent 循环 | `for bx, by in T.Persistent([...], sm_num, block_id)` |

### 同步和屏障

| 指令 | 用途 |
|------|------|
| `T.sync_threads()` | 线程块同步 |
| `T.ptx_wait_group(n)` | 等待异步拷贝 |
| `T.mbarrier_wait_parity(barrier, parity)` | MBarrier 等待 |
| `T.warpgroup_arrive()` / `T.warpgroup_commit_batch()` / `T.warpgroup_wait(n)` | WGMMA 同步 |

### 注解和优化

| 指令 | 用途 |
|------|------|
| `T.annotate_layout({buf: layout})` | 内存布局注解 |
| `T.use_swizzle(panel_size, enable)` | L2 光栅化 |
| `T.annotate_l2_hit_ratio(buf, ratio)` | L2 缓存提示 |

### Warp 操作

| 指令 | 用途 |
|------|------|
| `T.shfl_sync(value, src_lane)` | 广播 |
| `T.shfl_down(value, delta)` | 下移 |
| `T.shfl_xor(value, delta)` | XOR 交换 |
| `T.warp_reduce_sum/max` | Warp 归约 |
| `T.ballot(predicate)` | 投票 |

### 原子操作

| 指令 | 用途 |
|------|------|
| `T.atomic_add(dst, val)` | 原子加 |
| `T.atomic_max/min(dst, val)` | 原子最大/最小 |

## 11. 常见错误

1. **忘记 `T.clear(C_local)`** → 累积垃圾值导致错误
2. **`num_stages` 过大** → 超出共享内存限制
3. **`T.sync_threads()` 在条件分支中** → 死锁
4. **FP8 不使用 `transpose_B=True`** → 编译失败或错误结果
5. **`T.annotate_layout` 缺少 `make_mma_swizzle_layout` import** → bank conflicts
6. **`T.copy` 用于不兼容的 scope** → 编译失败
7. **Persistent kernel 中缺少边界检查 `bx * block_M < M`** → 越界访问
8. **Split-K 不使用原子操作合并结果** → 数据竞争

## 12. 性能调优 Checklist

- [ ] 分块大小: 从 `block_M=128, block_N=128, block_K=32` 开始
- [ ] 累积精度: 使用 `accum_dtype=T.float32`
- [ ] Pipeline: 设置 `num_stages=3`
- [ ] Swizzle: 添加 `T.annotate_layout` + `T.use_swizzle(panel_size=10)`
- [ ] 大矩阵 (M,N > 4096): 尝试 Persistent Kernel
- [ ] K >> M,N: 尝试 Split-K
- [ ] 最终调优: 使用 `AutoTuner` 搜索最佳配置
- [ ] 验证: `profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)`
