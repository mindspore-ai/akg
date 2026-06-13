### tl.range(start, end, step=1)
```python
for k0 in tl.range(0, K_TOTAL, BLOCK_K):
    ...
```
- **用途**: 在 kernel 内表达 runtime 边界的循环，常用于动态 K 轴、分块 reduction 或 grid-stride 处理。
- **约束**:
  - 循环体内仍要使用 mask 处理尾块和越界 lane。
  - 不要在需要 runtime 边界的场景使用 Python `range` 或长 `tl.static_range` 展开。
  - 若循环边界其实是很小的编译期常量，才考虑 `tl.static_range`。
