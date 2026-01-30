# 任务特征
**操作类型**：reduction，reduce所有轴；1D Tensor输入，标量输出  
**数据尺寸**：(4194304,) -> reduce轴极大  
**数据类型**：输入输出均为float32类型  
**任务特点**：数据规模很大，单个线程块难以覆盖全部元素，需要将reduce轴划分为多个块，并通过原子最小化完成跨块合并。

# 关键代码切片

## 优化1
```python
# 优化Triton
pid = tl.program_id(0)

for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    offsets = pid * BLOCK_SIZE + start + tl.arange(0, SUB_BLOCK_SIZE)
```
**优化内容**：kernel内，为了不超过硬件缓存，进行了二次切分（SUB_BLOCK_SIZE）。
**总结**：[通用优化] 当处理大规模数据时，通过调整grid设置使每个线程块处理适当多的数据，减少总线程块数量以降低调度开销，提升计算效率。但要注意不能超过硬件缓存，若可能超过硬件缓存，可以考虑二次切分

## 优化2
```python
# 简单Triton
row_min = float('inf')
for n_start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    # 加载数据……（步骤略）

    curr_min = tl.min(block_data)
    row_min = tl.minimum(curr_min, row_min)

# 优化Triton
curr_min = tl.full((SUB_BLOCK_SIZE,), float('inf'), dtype=tl.float32)
for start in range(0, BLOCK_SIZE, SUB_BLOCK_SIZE):
    # 加载数据……（步骤略）

    curr_min = tl.minimum(curr_min, block_data)
min_val = tl.min(curr_min)
```
**优化内容**：利用计算重组的思想，通过改变计算顺序和数据结构来提升性能。优化前每次循环都要对block_data进行行归约，归约操作相对耗时，重复次数较多时，严重影响性能，优化后利用curr_min保持矩阵结构，维护中间结果，减少归约次数，从而减少计算开销并提升性能。
**总结**：将通过申请额外的UB暂存中间结果，将多次归约合并为一次的归约，以分摊归约操作的开销，提升整体性能。​​

## 优化3
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：
# （当前AI core为40）

# 1. grid = 4194304 / 262144 = 16 < 40, UB用满 -> 性能：15.12 us
triton.Config({'BLOCK_SIZE': 262144, 'SUB_BLOCK_SIZE': 16384}),

# 2. grid = 4194304 / 131072 = 32 < 40, UB用满 -> 性能：9.61 us
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 16384}),

# 3. grid = 4194304 / 131072 = 32 < 40, UB未用满 -> 性能：10.29 us
triton.Config({'BLOCK_SIZE': 131072, 'SUB_BLOCK_SIZE': 8192}),

# 4. grid = 4194304 / 104858 = 40, UB用满, 带尾块 -> 性能：10.17 us
triton.Config({'BLOCK_SIZE': 104858, 'SUB_BLOCK_SIZE': 16384}),

# 5. grid = 4194304 / 65536 = 64 > 40, UB用满 -> 性能：11.64 us
triton.Config({'BLOCK_SIZE': 65536, 'SUB_BLOCK_SIZE': 32768}),
```
**优化内容**：在大规模数据场景下，测试结果显示，最优配置为配置2，特征为：网格数接近AI Core数量、UB用满，且无尾块。网格数过小会导致并行度不足，无法充分利用计算资源；网格数过大则会引入过多调度开销。同时，配置4网格数更接近AI Core数量但含有尾块，性能略差，说明数据对齐对性能有影响。此外，配置3结果显示UB未用满也会降低性能。
**总结**：处理大规模数据时，应在网格数接近AI Core数量、避免尾块和确保UB用满三者间找到平衡点。