---
name: tilelang-ascend-example-grouped-gemm
description: "分组/动态批次的 TileLang Ascend Expert 模式实现示例。当生成含分组/动态批次的算子时可参考此示例。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 分组矩阵乘法 — TileLang Ascend 实现示例（Expert 模式）

**编程模式**：Expert（手动管理 L1/L0C 内存层级）

**关键技术点**：
- **block_metadata 预计算表**：替代三维 Kernel，用 `[batch_idx, m_start, valid_rows]` 表描述每个 block 的归属
- **一维 Kernel + 手动索引分解**：`T.Kernel(total_m_blocks * n_num)` + `cid // n_num` / `cid % n_num`
- **静态循环边界**：`T.ceildiv(K, block_K)` 替代动态边界（TileLang Ascend 不支持循环次数依赖 tensor 值）

## Host 侧：block_metadata 预计算

```python
def construct_inputs(batch_sizes_list, K, N, block_M, device, dtype):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)

    A = torch.randn(batch_sum, K, device=device, dtype=dtype)
    B = torch.randn(batch_count, K, N, device=device, dtype=dtype)

    metadata_list = []
    current_global_offset = 0

    for batch_idx, size in enumerate(batch_sizes_list):
        num_blocks = (size + block_M - 1) // block_M
        for i in range(num_blocks):
            local_start = i * block_M
            m_start_global = current_global_offset + local_start
            valid_m = min(block_M, size - local_start)
            metadata_list.append([batch_idx, m_start_global, valid_m])
        current_global_offset += size

    block_metadata = torch.tensor(metadata_list, device=device, dtype=torch.int32)
    return A, B, block_metadata
```

## Kernel：分组 GEMM

```python
@tilelang.jit(out_idx=[2])
def grouped_gemm(batch_sizes_list, K, N, block_M, block_N, block_K, dtype="float16"):
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = "float32"
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
    n_num = (N + block_N - 1) // block_N

    @T.prim_func
    def kernel(
        A: T.Tensor([batch_sum, K], dtype),
        B: T.Tensor([batch_count, K, N], dtype),
        C: T.Tensor([batch_sum, N], dtype),
        # Metadata table: [batch_idx, m_start_offset, valid_rows]
        block_metadata: T.Tensor([total_m_blocks, 3], "int32"),
    ):
        with T.Kernel(total_m_blocks * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            cur_batch_idx = block_metadata[bx, 0]
            m_start = block_metadata[bx, 1]
            # Partial memory movement (tail handling) is not yet supported;
            # this variable is currently unused.
            _actual_rows = block_metadata[bx, 2]

            A_L1 = T.alloc_L1((block_M, block_K), dtype)
            B_L1 = T.alloc_L1((block_K, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, block_K)
                for k in T.serial(loop_k):
                    # Copyin
                    T.copy(
                        A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K],
                        A_L1,
                    )
                    T.copy(
                        B[
                            cur_batch_idx,
                            k * block_K : (k + 1) * block_K,
                            by * block_N : (by + 1) * block_N,
                        ],
                        B_L1,
                    )
                    T.barrier_all()

                    # Compute
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()

                # Copyout
                T.copy(
                    C_L0,
                    C[
                        m_start : m_start + block_M,
                        by * block_N : by * block_N + block_N,
                    ],
                )

    return kernel
```

**调用方式**：

```python
func = grouped_gemm(tuple([64, 128, 256]), 8192, 8192, 64, 64, 64)
A, B, block_metadata = construct_inputs([64, 128, 256], 8192, 8192, 64, device, dtype)
out = func(A, B, block_metadata)
```

**设计要点**：
- 静态循环边界 + 条件判断（替代动态边界）：`batch_sizes_list` 以 tuple 传入 `@jit` 层，在编译期展开为具体值，避免动态循环边界
- 预计算表（替代三维 Kernel）：`block_metadata` 由 host 预计算并作为 tensor 传入 kernel，替代 `T.Kernel` 的三维 block 数
- `m_start` 从 metadata 表读取，实现分组间不同起始偏移
- Kernel: `T.Kernel(total_blocks)` + 手动索引分解
