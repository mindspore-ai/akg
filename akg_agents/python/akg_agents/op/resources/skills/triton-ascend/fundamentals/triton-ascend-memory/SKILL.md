---
name: triton-ascend-memory
description: "Ascend NPU 内存访问优化策略，包括 UB（统一缓冲区）利用、数据布局优化、合并访存和预取技巧。适用于内存带宽受限、需要优化数据搬运效率、或处理大规模数据的内核代码性能优化场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 内存访问优化

## 块大小选择原理

块大小需要根据算子类型和硬件存储层级来平衡：

- **VEC 类算子**（element-wise、reduce、softmax 等）：数据需放入 UB（192KB/VEC），`BLOCK_SIZE * sizeof(dtype)` 需小于 UB 可用容量，同时兼顾计算并行度。过小并行度不足，过大溢出 UB
- **CUBE 类算子**（matmul、attention 等）：左矩阵放 L0A（* KB），右矩阵放 L0B（* KB），结果放 L0C（* KB），具体参考硬件信息文档：
  - `m0 * k0 * sizeof(A.dtype) ≤ * KB`（L0A）
  - `k0 * n0 * sizeof(B.dtype) ≤ * KB`（L0B）
  - `m0 * n0 * sizeof(C.dtype) ≤ * KB`（L0C）
- 所有数据传输按 **256 Bytes 对齐**，BLOCK_SIZE 为 32 的倍数最优

## 2D 数据：优先 tl.make_block_ptr

```python
A_block_ptr = tl.make_block_ptr(
    base=A_ptr, shape=(M, K), strides=(stride_am, stride_ak),
    offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
)
a = tl.load(A_block_ptr, boundary_check=(0, 1))
# 移动指针
A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
```

## 连续内存：一维访问

非连续张量先 `.contiguous()` 转换，再用一维 ptr + offsets 访问：

```python
class ModelNew(torch.nn.Module):
    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kernel[grid](x, out, n, BLOCK_SIZE=1024)
        return out
```

一维访问比 stride 计算效率更高，推荐优先使用。

## 对齐要求
- Ascend 256B 对齐: element-wise / reduce 算子
- Ascend 512B 对齐: MatMul 切分
- 数据搬运带宽上限约 256*256B，据此设计搬运策略

## 要点
- 优先 `.contiguous()` + 一维访问
- 连续内存访问效率远高于 stride 计算开销
