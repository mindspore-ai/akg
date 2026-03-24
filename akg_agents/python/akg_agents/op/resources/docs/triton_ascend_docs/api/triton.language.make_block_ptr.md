### tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)
```python
block_ptr = tl.make_block_ptr(
    base=ptr,                    # 基础指针
    shape=(M, N),                # 完整矩阵形状
    strides=(stride_m, stride_n), # 步长
    offsets=(start_m, start_n),   # 当前块偏移
    block_shape=(BLOCK_M, BLOCK_N), # 块形状
    order=(1, 0)                 # 内存布局顺序
)
```
- **参数**:
  - `base`: 基础内存指针
  - `shape`: 完整张量的形状
  - `strides`: 每个维度的步长
  - `offsets`: 当前块的起始偏移
  - `block_shape`: 当前块的大小
  - `order`: 内存布局顺序 (1, 0) 表示行主序
- **返回**: 块指针对象
- **用途**: 高效访问 2D 数据块

