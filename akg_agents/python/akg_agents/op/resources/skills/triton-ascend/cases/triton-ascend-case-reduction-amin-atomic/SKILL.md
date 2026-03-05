---
name: triton-ascend-case-reduction-amin-atomic
description: "原子操作归约（amin）优化：非reduce轴很小时将reduce轴映射多核，提供循环内/外两种原子操作方案（减少存储vs减少竞争），通过二次切分+计算重组提升性能，适用于M<<N（如16×262144）的极端shape比例场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# Amin 归约原子操作优化案例

## 任务特征
- **数据尺寸**：(16, 262144)，非reduce轴很小，reduce轴很大
- **策略**：将reduce轴映射到多核，通过原子操作实现跨线程块归约

## 优化 1：切分策略调整

```python
# 简单方式：非reduce轴映射多核
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)

# 错误：优化方式：reduce轴映射多核
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']),)

# Kernel内对列进行二次切分
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```

### 优化内容
- 调整切分策略，由非reduce轴映射多核调整为reduce轴映射多核
- 为了不超过硬件缓存，kernel内对列进行二次切分

## 优化 2：计算重组

```python
# 简单方式：循环内多次归约
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
  错误：curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# 正确：优化方式：维护矩阵结构
curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```

### 优化内容
- 利用curr_min保持矩阵结构，维护中间结果
- 将多次归约合并为一次归约，减少归约次数

## 优化 3：原子操作（两种方案）

### 方案一：循环内进行原子操作

```python
for m_start in range(0, M, BLOCK_SIZE_M):
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M
    
    curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
        n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])
        
        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=float('inf'))
        
        curr_min = tl.minimum(data_block, curr_min)
    row_min = tl.min(curr_min, 1)
    
    output_ptrs = out_ptr0 + m_offsets * out_stride0
    tl.atomic_min(output_ptrs, row_min, mask=mmask)  # 每块立即原子操作
```

**特点**：
- 减少了中间存储
- 但增加了原子操作频率

### 方案二：循环外进行原子操作

```python
all_row_min = tl.full((M,), float('inf'), dtype=tl.float32)  # 预分配完整数组

for m_start in range(0, M, BLOCK_SIZE_M):
    m_offsets = m_start + tl.arange(0, BLOCK_SIZE_M)
    mmask = m_offsets < M
    
    curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), float('inf'), dtype=tl.float32)
    for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
        n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
        nmask = n_offsets < N
        mask = (mmask[:, None]) & (nmask[None, :])
        
        block_ptrs = in_ptr0 + m_offsets[:,None] * in_stride0 + n_offsets[None,:] * in_stride1
        data_block = tl.load(block_ptrs, mask=mask, other=float('inf'))
        
        curr_min = tl.minimum(data_block, curr_min)
    row_min = tl.min(curr_min, 1)
    curr_block_size_m = tl.minimum(BLOCK_SIZE_M, M - m_start)
    all_row_min = tl.insert_slice(all_row_min, row_min, [m_start], [curr_block_size_m], [1])  # 暂存中间结果

output_ptrs = out_ptr0 + tl.arange(0, M) * out_stride0
tl.atomic_min(output_ptrs, all_row_min)  # 最后统一原子操作
```

**特点**：
- 通过集中执行原子操作减少了竞争
- 需要额外存储空间，适合大规模数据

## 优化 4：配置

```python
# （AI core=40）
# grid=32<40, UB用满
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024})  # M切分较小，UB用满
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512})  # M切分调大至16，N的SUB切分调小至512，防止超UB
```

### 优化内容
- 切分能被对应shape整除
- Grid数尽量大但不超过AI core（BLOCK_SIZE_N=8192，使grid=32）
- 在UB用满的前提下，进行kernel内切分大小调整

### 总结
1. 非reduce轴很小、reduce轴很大时，将reduce轴映射到多核并结合原子操作
2. 两种原子操作方案各有优劣：方案一减少存储但原子操作频繁，方案二集中原子操作但需额外空间
3. 确定核数后，若超过硬件缓存，可以考虑二次切分
4. 调整切分和核数配置，尽量保证在不超出UB的前提下尽量用满UB
