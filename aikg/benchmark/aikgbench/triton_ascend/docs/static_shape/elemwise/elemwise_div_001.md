# 任务特征
**操作类型**：broadcast类型，broadcast第一根轴；2D Tensor输入，2D Tensor输出
**数据尺寸**：(131072, 16)、(1, 16)，数据shape大，M维度较大
**数据类型**：float16
**任务特点**：操作类型为elementwise，可以向量化操作；triton kernel里面可以直接load一个向量进行单次操作；只需要对第一维（M维）进行广播；选择通过NUM_BLOCKS固定grid数切分M维度，在kernel内部通过for循环切分SUB_M进一步细分行处理，为double buffering和指令流水提供优化空间。由于divisor在M维度广播，只依赖N维度，可以在循环外加载一次然后复用，减少内存访问。

# 关键代码切片

## 优化1
```python
# 优化Triton切分配置：
# NUM_BLOCKS控制核数，SUB_M控制内部每次处理行数
triton.Config({'NUM_BLOCKS': 32, 'SUB_M': 512}),
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 512}), # 最优，核数=40，用满物理核
triton.Config({'NUM_BLOCKS': 64, 'SUB_M': 512}), # 核数>40，调度开销大
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 256}), # SUB_M偏小
triton.Config({'NUM_BLOCKS': 40, 'SUB_M': 1024}), # SUB_M较大
grid = lambda meta: (meta['NUM_BLOCKS'],)
```
**优化内容**：通过grid切分M维度（grid = NUM_BLOCKS），控制核数≤40。对于M=131072，NUM_BLOCKS=40时充分利用物理核数。内部通过SUB_M参数控制每次处理的行数，SUB_M=512时在UB利用率和寄存器压力之间达到平衡，性能最优。每个核负责rows_per_block = (M + NUM_BLOCKS - 1) // NUM_BLOCKS行。

**总结**：[通用优化] 在Ascend平台上，对于某一维度较大的2D数据，通过固定NUM_BLOCKS≤40实现核间并行，核内参数切分控制数据粒度以平衡UB利用率和寄存器压力。

## 优化2
```python
# N维度较小(N=16)，完整处理不切分
offs_n = tl.arange(0, N)  # N=16

# divisor所有行共享，在循环外加载一次
divisor = tl.load(divisor_ptr + offs_n)  # shape: (N,)

# 内层循环：每次处理SUB_M行
for sub_start in range(row_start, row_end, SUB_M):
    offs_m = sub_start + tl.arange(0, SUB_M)
    
    # 加载dividend: (SUB_M, N)
    dividend = tl.load(dividend_ptr + dividend_offs, mask=mask_2d, other=0.0)
    
    # divisor广播: (N,) -> (SUB_M, N)
    output = dividend / divisor
```
**优化内容**：由于N维度较小(N=16)，选择完整处理不切分，最大化UB利用率。divisor所有行共享(shape为(1, N))，在循环外加载一次，然后在循环内自动广播到每个(SUB_M, N)块复用，避免重复加载。N维度不切分既减少了循环开销，又使得divisor可以一次性加载完整复用。

**总结**：[通用优化] 在Ascend平台上，对于较小的维度，应完整处理不切分以最大化UB利用率；维度大小决定是否切分，而非广播方向。

