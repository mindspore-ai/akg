# 任务特征
**操作类型**：reduction+elemetwise融合，reduce轴为最后一根轴。3个输入均为3D Tensor，3个输出分别为3D Tensor, 3D Tensor, 2D Tensor
**数据尺寸**：(16, 1024, 2048), (16, 1024, 2048), (16, 1024, 2048) -> 算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为​逐元素与归约​计算​融合​，先进行向量化逐元素操作，再沿列方向求和归约；逐元素操作与归约在核内融合，避免中间结果写回全局内存，提升计算效率；行方向操作可以向量化并行处理，列方向维度较大，可能超出硬件缓存，也需进行切分，因此通过​​行列双向分块​优化内存访问细化并行粒度，利用for循环处理行列分块；采用​​Auto-Tuning机制​​动态选择分块配置，平衡计算负载与内存占用。

# 关键代码切片

## 优化1
```python
# 优化Triton
x_reshaped = x.reshape(BM, N)
weight_reshaped = weight.reshape(BM, N)
grad_reshaped = grad.reshape(BM, N)

weighted_x_reshaped = weighted_x.reshape(BM, N)
grad_weight_reshaped = grad_weight.reshape(BM, N)
grad_x_reshaped = grad_x.reshape(BM)
```
**优化内容​**​：通过将输入张量的前两个维度(B, M)合并为单个维度BM，将三维张量重塑为二维张量，从而将二维网格并行策略简化为一维网格，减少了索引计算的复杂性，优化了内存访问模式，提高了内核执行效率。
**​​总结**​​：[通用优化] 将高维张量重塑为低维张量可以简化并行策略，优化内存访问模式，提升内核性能。

## 优化2
```python
# 优化Triton
for bm_start in range(0, BLOCK_SIZE_BM, SUB_BLOCK_SIZE_BM):
    bm_offsets = pid * BLOCK_SIZE_BM + bm_start + tl.arange(0, SUB_BLOCK_SIZE_BM)
    bm_mask = bm_offsets < BM
```
**优化内容**：
调整行切分策略：
- 由每个kernel计算一行调整为计算多行（BLOCK_SIZE_BM行），以此减少总线程块数量；
- kernel内，为了不超过硬件缓存，对行进行了二次切分（SUB_BLOCK_SIZE_BM）。
**总结**：[通用优化] 当处理大规模数据时，通过调整grid设置使每个线程块处理适当多的数据，减少总线程块数量以降低调度开销，提升计算效率。但要注意不能超过硬件缓存，若可能超过硬件缓存，可以考虑二次切分

## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. grid数16 * 1024 / 32 = 512 > 40，reduce轴切分值较小，UB占满 -> 性能：1105.84 us
triton.Config({'BLOCK_SIZE_BM': 32, 'SUB_BLOCK_SIZE_BM': 32, 'BLOCK_SIZE_N': 128}),

# 2. grid数1024 > 40，reduce轴切分值增至256，BLOCK_SIZE_M调小为16，防止超出UB -> 性能：1110.47 us
triton.Config({'BLOCK_SIZE_BM': 16, 'SUB_BLOCK_SIZE_BM': 16, 'BLOCK_SIZE_N': 256}),

# 3. grid数2048 > 40，reduce轴切分值增至512，BLOCK_SIZE_M调小为8，防止超出UB -> 性能：1091.26 us
triton.Config({'BLOCK_SIZE_BM': 8, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),

# 4. grid数32 < 40，reduce轴切分值较大，UB占满 -> 性能：1098.53 us
triton.Config({'BLOCK_SIZE_BM': 512, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),

# 5. grid数40 = 40，有尾块，reduce轴切分值较大，UB占满 -> 性能：1094.60 us
triton.Config({'BLOCK_SIZE_BM': 416, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),
```
**优化内容**：
- 通过autotune提供多样化块大小配置，考虑reduce轴分块大小、UB是否占满等多重因素。性能测试显示，reduce轴切分值较大，且占满UB时性能较好，如配置3、4和5。并且grid数较大时，性能最优。
**总结**：针对大shape算子，分块策略应在优先占满UB的前提下为reduce轴分配较大切分尺寸——该策略旨在减少循环次数，但代价是单次迭代计算负载和归约计算负载增加。因此，为reduce轴设置较大切分值的同时也需做好权衡，以达成整体性能最优。此外，将grid数调大有可能会使性能提升。