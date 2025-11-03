# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(1000, 8192) -> 算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴的reduce，每行的reduce操作可以向量化并行处理；可以通过​​行列双向分块​​（BLOCK_SIZE_M和BLOCK_SIZE_N）优化内存访问，利用for循环实现列方向分块加载；采用​​Auto-Tuning机制​​动态选择最优分块配置，平衡并行粒度与内存占用；各行计算完全独立，无需原子操作，输出直接存储。

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
**总结**：[通用优化] 当处理大规模数据时，通过调整grid设置使每个线程块处理适当多的数据，减少总线程块数量以降低调度开销，提升计算效率。但要注意不能超过硬件缓存，若可能超过硬件缓存，可以考虑二次切分

## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. grid < 40 -> 性能：28.64 us
triton.Config({'BLOCK_SIZE_M': 50, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# 2. grid = 40，SUB切分含尾块 -> 性能：16.54 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096})

# 3. grid = 40，SUB切分不含尾块 -> 性能：16.00 us
triton.Config({'BLOCK_SIZE_M': 25, 'SUB_BLOCK_SIZE_M': 25, 'BLOCK_SIZE_N': 512})

# 4. grid > 40，且非核数整数倍 -> 性能：25.86 us
triton.Config({'BLOCK_SIZE_M': 20, 'SUB_BLOCK_SIZE_M': 20, 'BLOCK_SIZE_N': 512})
```
**优化内容**：
- 通过autotune提供多样化块大小配置，考虑grid与核数的关系、是否存在尾块等多重因素，最大化硬件利用率。
- 性能测试显示，当前最优切分为配置3，其特征为grid等于核数，SUB切分不含尾块。配置1问题在于数据规模较大的情况下核数未用满；配置2用满核数，核内内计算中，for循环中计算过程执行 $\lceil 25/4 \rceil \times 8192/4096=14$，尽管比配置1的16小，但其最后一个循环为尾块特殊情况（$25 \equiv 1 \pmod{4}$），所以性能略逊一筹；配置4的问题在于：grid数量超出核数且非核数整数倍，各核计算任务不均匀，导致性能较差。
**总结**：
- [通用优化] 尾块计算会降低算子性能，计算循环次数相似时，优先采用不含尾块的切分方案。
- 处理大规模数据时，为提升性能，grid数量应尽量等于AI core核数，最大化硬件利用率，并使各核计算任务分配均匀。