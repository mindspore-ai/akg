# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(65536, 2048) -> 非reduce轴非常大，reduce轴中等
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴的reduce，reduce轴的每一步计算都依赖于前一步的结果，所以常规情况下不能并行。该任务的非reduce轴并行处理，reduce轴中等，进行向量化，由于UB大小有限，需要对reduce轴进行切分。

# 关键代码切片

## 优化1
```python
# 简单Triton
total_sum = 0.0
for n_offset in range(0, N, BLOCK_SIZE):
    # elemwise operatrions...

    row_sum += tl.sum(block_vals)

# 优化Triton
acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    # elemwise operatrions...
    
    row_sum += block_vals
row_sum = tl.sum(row_sum, axis=1)
```
**优化内容**：利用计算重组的思想，通过改变计算顺序和数据结构来提升性能。优化前每次循环都要对block_vals进行行规约，规约操作相对耗时，重复次数较多时，严重影响性能，优化后利用row_sum保持矩阵结构，维护中间结果，减少规约次数，从而减少计算开销并提升性能。
**总结**：将通过申请额外的UB暂存中间结果，将多次规约合并为一次的归约，以分摊归约操作的开销，提升整体性能。​​

## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：

# 1. reduce轴切分值较小，UB占满 -> 性能：700.42 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. reduce轴切分值增至512，BLOCK_SIZE_M调小为32，防止超出UB -> 性能：695.08 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512}),

# 3. reduce轴切分值增至1024，BLOCK_SIZE_M调小为16，防止超出UB -> 性能：685.65 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 1024}),

# 4. reduce轴切分值增至2048，BLOCK_SIZE_M调小为8，防止超出UB -> 性能：686.89 us
triton.Config({'BLOCK_SIZE_M': 8, 'BLOCK_SIZE_N': 2048}),

# 5. reduce轴切分值较大，UB未占满 -> 性能：743.83 us
triton.Config({'BLOCK_SIZE_M': 4, 'BLOCK_SIZE_N': 2048}),
```
**优化内容**：
- 通过autotune提供多样化块大小配置，考虑reduce轴分块大小、UB是否占满等多重因素。
- 性能测试显示，当前较好的切分为配置3和4，最优为配置3，共同特征为：reduce轴切分值较大，且占满UB。配置1和2的轴切分值较小，导致性能略差。而配置5未占满UB，性能显著低于其他占满UB的配置。
**总结**：针对大shape算子，分块策略应在优先占满UB的前提下为reduce轴分配较大切分尺寸——该策略旨在减少循环次数，但代价是单次迭代计算负载和归约计算负载增加。因此，为reduce轴设置较大切分值的同时也需做好权衡，以达成整体性能最优。