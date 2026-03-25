### tl.extra.cann.extension.extract_slice(ful, offsets, sizes, strides)
```python
sub_tensor = tl.extra.cann.extension.extract_slice(tensor, [0], [32], [1])
```
- **作用**: 从输入张量中按照偏移量、大小和步幅提取一个切片。
- **参数**:
  - `ful`: 要提取切片的源张量
  - `offsets`: 切片在各维上的起始偏移
  - `sizes`: 切片在各维上的大小
  - `strides`: 切片在各维上的步长
- **返回**: 提取后的子张量
