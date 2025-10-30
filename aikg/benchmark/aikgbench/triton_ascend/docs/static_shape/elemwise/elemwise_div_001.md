# 任务特征
**操作类型**：broadcast类型，broadcast第一根轴；2D Tensor输入，2D Tensor输出
**数据尺寸**：(1024, 1024)、(1, 1024)，数据shape较小
**数据类型**：float16
**任务特点**：操作类型为elementwise，可以向量化操作；triton kernel里面可以直接load一个向量进行单词操作；需要对第一维进行广播；第二维不大，选择切分第一维，将多行分配给每个线程块，并在 kernel 内部通过 for 循环分块处理，既保证了内存访问的连续性，又为 double buffering 和指令流水提供了优化空间。同时，将 grid size 显式限制在物理核数以内，以匹配硬件并行能力，提升整体执行效率。

# 关键代码切片

## 优化1
```python
# 初始Triton切分配置：
BLOCK_M = 16
grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

# 优化Triton切分配置：
'NUM_BLOCKS': 32, 'SUB_M': 8, # 最优配置，shape是2的幂次倍数
'NUM_BLOCKS': 40, 'SUB_M': 8, # 最优配置，等于核数，shape是核数的整数倍
grid = lambda meta: (meta['NUM_BLOCKS'],)
```
**优化内容**：通过设置grid大小小于等于物理核数，降低调度开销

**总结**：[通用优化] 在Ascend平台上，当triton kernel的grid数较高时，可以调整NUM_BLOCKS设置，使得其降低至真实物理核（AI Vector）数，使得 kernel 启动时，grid 数在硬件约束范围内调整，实现性能优化。

## 优化2
```python
# 简单Triton内核实现：每次搬运整个BLOCK_M行
offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
mask_m = offs_m < M
# 优化Triton内核实现：将BLOCK_M行分块，每次搬运SUB_M行
for sub_start in range(row_start, row_end, SUB_M):
    # 当前子块的行索引
    offs_m = sub_start + tl.arange(0, SUB_M)
    mask_m = offs_m < row_end
```
**优化内容**：triton 内核部分使用for循环，尝试提高UB的利用率，提高并行度。

**总结**：[通用优化] 在Ascend设备上，Triton 内核可以将数据分块，通过添加for循环，开启二次切分，提高UB的利用率。
