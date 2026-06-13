### tl.arange(start, end)
```python
offsets = tl.arange(0, BLOCK_SIZE)
```
- **参数**: `start`, `end` - 起始和结束值
- **返回**: 连续整数序列
- **用途**: 创建索引序列
- **约束**:
  - `start` 和 `end` 应是编译期可确定值，`end - start` 通常需要是 2 的幂。
  - 对非 2 的幂或动态真实长度，使用 padded `BLOCK_SIZE` 创建 offsets，再用 `offsets < logical_size` 做 mask。
  - 不要写 `tl.arange(0, N)`，其中 `N` 是普通 runtime 参数。
