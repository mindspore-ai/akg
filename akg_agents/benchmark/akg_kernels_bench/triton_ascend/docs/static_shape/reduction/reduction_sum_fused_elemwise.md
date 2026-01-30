# 任务特征
**操作类型**：reduction+elementwise融合，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(1000, 8192), (8192,)，非reduce轴中等，reduce轴较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为​逐元素与归约​计算​融合​，先进行向量化逐元素操作，再沿列方向求和归约。整体优化逻辑以reduce算子为主，非reduce轴并行处理，reduce轴较大，进行向量化，由于UB大小有限，需要对reduce轴进行切分。

# 关键代码切片

## 优化1
```python
# 优化Triton
pid = tl.program_id(0)
for m_start in range(0, BLOCK_SIZE_M, SUB_BLOCK_SIZE_M):
    m_offsets = pid * BLOCK_SIZE_M + m_start + tl.arange(0, SUB_BLOCK_SIZE_M)
```
**优化内容**：
调整行切分策略：
- 由每个kernel计算一行调整为计算多行（BLOCK_SIZE_M行），以此减少总线程块数量；
- kernel内，为了不超过硬件缓存，对行进行了二次切分（SUB_BLOCK_SIZE_M）。
**总结**：当处理大规模数据时，通过调整grid设置使每个线程块处理适当多的数据，减少总线程块数量以降低调度开销，提升计算效率。但要注意不能超过硬件缓存，若可能超过硬件缓存，可以考虑二次切分

## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. grid = 1000 / 50 = 20 < 40-> 性能：91.69 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# 2. grid = 40，SUB切分含尾块 -> 性能：53.30 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048})

# 3. grid = 40，SUB切分不含尾块 -> 性能：47.58 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 256})

# 4. grid > 40，且非核数整数倍 -> 性能：79.00 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 256})
```
**优化内容**：
- 通过autotune提供多样化块大小配置，考虑grid与核数的关系、是否存在尾块等多重因素，最大化硬件利用率。
- 性能测试显示，当前最优切分为配置3，其特征为grid等于核数，SUB切分不含尾块。配置1问题在于数据规模较大的情况下核数未用满；配置2用满核数，核内内计算中，for循环中计算过程执行 $\lceil 25/4 \rceil \times 8192/2048=28$，尽管比配置1的32小，但其最后一个循环为尾块特殊情况（$25 \equiv 1 \pmod{4}$），所以性能略逊一筹；配置4的问题在于：grid数量超出核数且非核数整数倍，各核计算任务不均匀，导致性能较差。
**总结**：
- 尾块计算会降低算子性能，计算循环次数相似时，优先采用不含尾块的切分方案。
- 处理大规模数据时，为提升性能，grid数量应尽量等于AI core核数，最大化硬件利用率，并使各核计算任务分配均匀。

## 优化3
```python
# 简单Triton
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    # elemwise operatrions...

    total_sum += tl.sum(tl.where(mask, t3, 0.0))

# 优化Triton
acc = tl.zeros([SUB_BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    # elemwise operatrions...
    
    acc += tl.where(mask, t3, 0.0)
total_sum = tl.sum(acc, axis=1)
```
**优化内容**：利用计算重组的思想，通过改变计算顺序和数据结构来提升性能。优化前每次循环都要进行行归约，归约操作相对耗时，重复次数较多时，严重影响性能，优化后利用acc保持矩阵结构，维护中间结果，减少归约次数，从而减少计算开销并提升性能。
**总结**：将通过申请额外的UB暂存中间结果，将多次归约合并为一次的归约，以分摊归约操作的开销，提升整体性能。​​
