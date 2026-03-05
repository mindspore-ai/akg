---
name: triton-ascend-case-reduction-amax-large
description: "非reduce轴很小、reduce轴很大的归约优化：将reduce轴映射到多核（而非常规的非reduce轴），使用原子操作跨线程块归约，通过二次切分避免超UB，适用于极端shape比例（M<<N如16×262144）的归约场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 大规模 Amax 归约优化（reduce轴映射多核）

## 任务特征
- **数据尺寸**：(16, 262144)，非reduce轴很小，reduce轴很大
- **策略**：将reduce轴映射到多核，使用原子操作

## 优化 1：切分策略调整

```python
# 错误：简单方式：非reduce轴映射多核
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# 正确：优化方式：reduce轴映射多核
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Kernel内对列进行二次切分
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```

## 优化 2：原子操作

### 方案一：循环内原子操作
```python
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    tl.atomic_min(output_ptrs, row_min, mask=mmask)
```

### 方案二：循环外原子操作
```python
all_row_min = tl.full((M,), float('inf'), dtype=tl.float32)
for m_start in range(0, M, BLOCK_SIZE_M):
    row_min = tl.min(curr_min, 1)
    all_row_min = tl.insert_slice(all_row_min, row_min, ...)
tl.atomic_min(output_ptrs, all_row_min)
```

## 优化 3：配置

```python
# grid=32<40, UB用满
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024})
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512})
```

### 总结
非reduce轴很小、reduce轴很大时，将reduce轴映射到多核并结合原子操作，通过二次切分避免超出UB。
