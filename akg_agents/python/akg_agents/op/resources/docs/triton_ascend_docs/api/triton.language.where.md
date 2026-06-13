### tl.where(condition, x, y)
```python
result = tl.where(mask, data, 0.0)
```
- **参数**:
  - `condition`: 条件张量
  - `x`, `y`: 选择值
- **返回**: 根据条件选择的值
- **用途**: SIMD 友好的条件选择
- **Ascend 约束**:
  - `tl.where` 适合选择数值，不要依赖它构造可能越界的 pointer 后再 load/store。
  - 对内存访问边界，应优先把条件合进 `tl.load` / `tl.store` 的 `mask`。
