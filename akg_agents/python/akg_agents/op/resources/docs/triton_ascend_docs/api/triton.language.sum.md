### tl.sum(x, axis)
```python
block_sum = tl.sum(data, axis=0)
```
- **参数**:
  - `x`: 输入张量
  - `axis`: 归约轴
- **返回**: 归约结果
- **约束**: 被归约轴必须是当前 block tensor 的静态维度；越界 lane 应在 load 时用 `other=0.0` 清零。
