# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(16, 262144) -> 非reduce轴很小，reduce轴很大，整体算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴，非reduce轴很小，reduce轴很大；可以采用列向并行策略，将reduce轴映射到多核，通过atomic_min实现跨线程块的最小值归约；可以使用行列双向分块优化内存访问，并切由于reduce轴很大，可以引入子列分块细化计算粒度，平衡并行效率与归约开销。

# 关键代码切片

## 优化1
```python
# 简单Triton
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), )

for n_start in range(0, N, BLOCK_SIZE_N):
    n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)

# 优化Triton
grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )

for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    n_offsets = pid * BLOCK_SIZE_N + n_start + tl.arange(0, SUB_BLOCK_SIZE_N)
```
**优化内容**：调整切分策略，由非reduce轴映射到多核调整为reduce轴映射多核。并且，为了不超过硬件缓存，kernel内对列进行了二次切分。
**总结**：[通用优化] reduce轴非常大时，可以考虑将reduce轴映射到多核并原子操作计算。当处理大规模数据时，通过调整grid设置使每个线程块处理适当多的数据，减少总线程块数量以降低调度开销，提升计算效率。但要注意不能超过硬件缓存，若可能超过硬件缓存，可以考虑二次切分

## 优化2
```python
# 简单Triton
row_min = -float('inf')
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    # 加载数据……（步骤略）

    curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# 优化Triton
curr_min = tl.full((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
    # 加载数据……（步骤略）
    
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```
**优化内容**：通过引入累加器矩阵延迟归约操作，将循环内的多次小归约合并为循环后的一次批量归约，从而减少计算开销并提升性能。
**总结**：将多次细粒度的归约操作，通过暂存中间结果合并为一次粗粒度的归约，以分摊归约操作的开销，提升整体性能。​​

## 优化3
```python
# 使能原子操作

# 优化方案一：循环内进行原子操作
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

# 优化方案二：循环外进行原子操作
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
**优化内容**：提供使能原子操作的两种方案。方案一在循环内进行原子操作，减少了中间存储但增加了原子操作频率；方案二在循环外进行原子操作，通过集中执行原子操作减少了竞争，但需要额外存储空间，适合大规模数据。两种方案均比不使用原子操作的方案在并行效率上有显著提升。两种方案各有优劣，需根据具体场景权衡。
**总结**：在非reduce轴很小，reduce轴很大时，原子操作可有效优化并行归约操作，带来算子性能提升。实际选择原子操作归约时，应根据数据规模和硬件特性权衡操作频率与存储开销，在竞争程度和内存使用间取得最佳平衡。

## 优化4
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：

# （当前AI core为40）
# grid数262144 / 8192 = 32 < 40, UB用满
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 1024}), # M切分较小，UB用满
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 8192, 'SUB_BLOCK_SIZE_N': 512}), # M切分调大至16，N的SUB切分调小至512，防止超UB
```
**优化内容**：切分能被对应shape整除、grid数尽量大但不超过AI core时，算子性能较好，因此BLOCK_SIZE_N都设为8192。在此基础上，且在UB用满的前提下，进行kernel内切分大小的调整，通过Autotune配置各种块大小。
**总结**：切分能被对应shape整除、grid数尽量大但不超过AI core时可能使性能达到最优，但实际应用中需结合具体任务特性进行针对性调优。同时，在不超出UB的前提下尽量用满UB。