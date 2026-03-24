### tl.cumprod(input, axis=0, reverse=False)
```python
cumulative_prod = tl.cumprod(data, axis=0)
reverse_cumprod = tl.cumprod(data, axis=1, reverse=True)
```
- **参数**:
  - `input`: 输入张量
  - `axis`: 累积乘积的轴 (默认为 0)
  - `reverse`: 是否反向累积 (默认为 False)
- **返回**: 累积乘积结果张量
- **用途**: 计算沿指定轴的累积乘积，常用于概率计算和序列处理

