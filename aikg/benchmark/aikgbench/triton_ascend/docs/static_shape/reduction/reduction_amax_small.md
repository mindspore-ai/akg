# 任务特征
**操作类型**：reduction，reduce轴为最后一根轴；2D Tensor输入，1D Tensor输出
**数据尺寸**：(16, 16) -> 非reduce轴和reduce轴均较小
**数据类型**：输入输出均为float32类型
**任务特点**：操作类型为归约最后一根轴的reduce，reduce轴的每一步计算都依赖于前一步的结果，所以常规情况下不能并行。该任务的非reduce轴和reduce轴均较小，所以不需要考虑过多的并行。

# 关键代码切片

## 优化1
```python
# 优化Triton——通过autotune测试了多种块大小配置，
# 配置及其性能如下：

# 1. 单核处理 -> 性能：2.16 us
triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16})

# 2. 多核并行处理 -> 性能：3.51 us
triton.Config({'BLOCK_SIZE_M': 1, 'BLOCK_SIZE_N': 16}),
```
**优化内容**：在算子规模很小时，单核（少核）处理比多核并行处理性能更优，说明并行化带来的开销超过了其收益（多核启动时间大于计算时间）。

## 总结
对于小规模计算任务，调小核数性能较优