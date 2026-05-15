### tl.swizzle2d(i, j, size_i, size_j, group_size)
```python
task_i, task_j = tl.swizzle2d(block_i, block_j, NUM_BLOCKS_I, NUM_BLOCKS_J, GROUP_SIZE)
```
- **参数**:
  - `i`, `j`: 原始块索引
  - `size_i`, `size_j`: 总块数
  - `group_size`: 分组大小(通常为2/4/8)
- **返回**: 重排后的块索引 (task_i, task_j)
- **用途**: 2D块重排,提升缓存局部性
- **适用场景**: 矩阵乘法等多维块计算,改善数据复用
- **注意**: 仅支持行优先(i方向)分组,列优先需手动实现
