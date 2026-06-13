### tl.expand_dims(input, axis)
```python
x_col = tl.expand_dims(x, 1)
```
- **参数**:
  - `input`: 输入 block tensor
  - `axis`: 新增维度位置
- **用途**: 显式构造广播维度，例如把 `(BLOCK_M,)` 变成 `(BLOCK_M, 1)`。
- **约束**:
  - `axis` 应是编译期可确定的整数。
  - 也可以使用 `x[:, None]` / `x[None, :]` 形成广播维度，但不要使用不支持的复杂 Python tensor indexing。
