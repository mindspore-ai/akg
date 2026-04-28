### tl.cumsum(input, axis=0, reverse=False, dtype=None)
```python
cumulative_sum = tl.cumsum(data, axis=0)
reverse_cumsum = tl.cumsum(data, axis=1, reverse=True)
```
- **参数**:
  - `input`: 输入张量
  - `axis`: 累积求和的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
  - `dtype`: 输出数据类型 (可选，默认与输入相同)
- **返回**: 累积求和结果张量
- **用途**: 计算沿指定轴的累积和，常用于前缀和计算

