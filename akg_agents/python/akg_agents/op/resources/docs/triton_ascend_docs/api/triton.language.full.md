### tl.full(shape, value, dtype)
```python
ones = tl.full((M, N), 1.0, dtype=tl.float32)
```
- **参数**:
  - `shape`: 张量形状
  - `value`: 填充值
  - `dtype`: 数据类型
- **返回**: 填充指定值的张量
- **约束**: `shape` 必须是编译期可确定 tuple；用于 reduction identity 时要选择正确初值。
