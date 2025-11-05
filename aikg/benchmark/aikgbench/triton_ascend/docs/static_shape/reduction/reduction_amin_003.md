# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(2048, 262144) -> 非reduce轴中等，reduce轴很大，整体算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴，非reduce轴中等，reduce轴很大；每行的reduce操作可以向量化并行处理；可以通过​​行列双向分块​​（BLOCK_SIZE_M和BLOCK_SIZE_N）优化内存访问，利用for循环实现列方向分块加载；采用​​Auto-Tuning机制​​动态选择最优分块配置，平衡并行粒度与内存占用；各行计算完全独立，无需原子操作，输出直接存储。

# 关键代码切片

## 优化1
```python
# 简单Triton
row_min = -float('inf')
for n_start in range(0, N, BLOCK_SIZE_N):
    # 加载数据……（步骤略）

    curr_min = tl.min(data_block, 1)
    row_min = tl.minimum(curr_min, row_min)

# 优化Triton
curr_min = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    # 加载数据……（步骤略）
    
    curr_min = tl.minimum(data_block, curr_min)
row_min = tl.min(curr_min, 1)
```
**优化内容**：通过引入累加器矩阵延迟归约操作，将循环内的多次小归约合并为循环后的一次批量归约，从而减少计算开销并提升性能。
**总结**：将多次细粒度的归约操作，通过暂存中间结果合并为一次粗粒度的归约，以分摊归约操作的开销，提升整体性能。​​

## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. reduce轴切分值较大, UB用满 -> 性能：2864.90 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048}), 

# 2. reduce轴切分值调大至4096，非reduce轴切分对应调小至4，防止超UB -> 性能：2840.48 us
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 4096}),

# 3. reduce轴切分值调大至8192，非reduce轴切分对应调小至2，防止超UB -> 性能：2801.20 us
triton.Config({'BLOCK_SIZE_M': 2, 'BLOCK_SIZE_N': 8192}),

# 4. reduce轴切分值调大至16384，非reduce轴切分对应调小至1，防止超UB -> 性能：2779.78 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16384}),
```
**优化内容**：通过autotune提供多样化块大小配置，UB均占满，reduce轴切分值较大时性能更优。
**总结**：针对大shape算子，分块策略应在优先占满UB的前提下为reduce轴分配较大切分尺寸——该策略旨在减少循环次数，但代价是单次迭代计算负载和归约计算负载增加。因此，为reduce轴设置较大切分值的同时也需做好权衡，以达成整体性能最优。