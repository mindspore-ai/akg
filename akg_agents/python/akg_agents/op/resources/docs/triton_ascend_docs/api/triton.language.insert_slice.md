### tl.insert_slice(ful, sub, offsets, sizes, strides)
```python
output = tl.insert_slice(output, output_sub, [offset], [size], [1])
```
- **作用**: 将子张量按照偏移量、大小和步幅插入到目标张量的指定位置。
- **参数**:
  - `ful`: 接收插入结果的目标张量
  - `sub`: 要插入的子张量
  - `offsets`: 插入区域在各维上的起始偏移
  - `sizes`: 插入区域在各维上的大小
  - `strides`: 插入区域在各维上的步长
- **返回**: 插入子张量后的新张量
