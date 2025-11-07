# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(2048, 8192) -> 非reduce轴为中等，reduce轴较大
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴的reduce，reduce轴的每一步计算都依赖于前一步的结果，所以常规情况下不能并行。该任务的非reduce轴为中等大小，reduce轴较大，由于UB大小有限，需要对reduce轴进行切分。

# 关键代码切片

## 优化1
```python
# 简单Triton
row_max = -float('inf')
for n_offset in range(0, N, BLOCK_SIZE):
    # 加载数据……（步骤略）

    curr_max = tl.max(data_block, 1)
    row_max = tl.maximum(curr_max, row_max)

# 优化Triton
curr_max = tl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), -float('inf'), dtype=tl.float32)
for n_start in range(0, N, BLOCK_SIZE_N):
    # 加载数据……（步骤略）
    
    curr_max = tl.maximum(data_block, curr_max)
row_max = tl.max(curr_max, 1)
```
**优化内容**：利用计算重组的思想，通过改变计算顺序和数据结构来提升性能。优化前每次循环都要对data_block进行行规约，规约操作相对耗时，重复次数较多时，严重影响性能，优化后利用curr_max保持矩阵结构，维护中间结果，减少规约次数，从而减少计算开销并提升性能。


## 优化2
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. grid = 2048 / 64 = 32 < 40, UB用满 -> 性能：29.05 us
triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256})

# 2. grid > 40，BLOCK_SIZE_N调整为512使UB用满 -> 性能：29.09 us
triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 512})

# 3. grid = 40，BLOCK_SIZE_N=512时超出UB，调整为256 -> 性能：25.73 us
triton.Config({'BLOCK_SIZE_M': 52, 'BLOCK_SIZE_N': 256})
```
**优化内容**：通过autotune测试不同块大小配置发现：grid等于核数时性能最优。但这一规律并非绝对，实际最优配置可能因具体计算任务特性和硬件状态而有所变化。

## 总结：
1. 将通过申请额外的UB暂存中间结果，将多次规约合并为一次的归约，以分摊归约操作的开销，提升整体性能。​​
2. 调整切分和核数配置，尽量保证在不超出UB的前提下尽量用满UB，并且尽量占满核数（可以考虑设被shape整除后的值或者物理核数的整数倍），实际应用中需结合具体任务特性在核数和切分配置中平衡。