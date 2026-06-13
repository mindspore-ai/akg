### tl.min(x, axis)
```python
min_val = tl.min(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 最小值
- **约束**: 越界或无效 lane 应在 load 时使用 `other=float("inf")`，不要先越界 load 再用 `tl.where` 修正。
