# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(2048, 262144) -> 非reduce轴中等，reduce轴很大，整体算子规格较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴的reduce，reduce轴的每一步计算都依赖于前一步的结果，所以常规情况下不能并行。该任务的非reduce轴中等，reduce轴很大，由于UB大小有限，需要对reduce轴进行切分，但是因为非reduce轴中等大小，所以不需要将reduce轴映射到多核。

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
**优化内容**：利用计算重组的思想，通过改变计算顺序和数据结构来提升性能。优化前每次循环都要对data_block进行行规约，规约操作相对耗时，重复次数较多时，严重影响性能，优化后利用curr_min保持矩阵结构，维护中间结果，减少规约次数，从而减少计算开销并提升性能。
**总结**：将通过申请额外的UB暂存中间结果，将多次规约合并为一次的归约，以分摊归约操作的开销，提升整体性能。​​

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
**总结**：针对reduce轴较大场景，分块策略应在优先占满UB的前提下为reduce轴分配较大切分尺寸——该策略旨在减少循环次数，但代价是单次迭代计算负载和归约计算负载增加。因此，为reduce轴设置较大切分值的同时也需做好权衡，以达成整体性能最优。