### tl.max(x, axis)
```python
max_val = tl.max(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最大值
- **约束**: 越界或无效 lane 应在 load 时使用 `other=-float("inf")`，不要先越界 load 再用 `tl.where` 修正。

