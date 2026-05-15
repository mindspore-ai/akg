### tl.dot(a, b, acc=None, allow_tf32=True)
```python
result = tl.dot(a, b, acc=accumulator)
```
- **参数**:
  - `a`, `b`: 输入矩阵
  - `acc`: 累加器 (可选)
  - `allow_tf32`: 是否允许 TF32 精度
- **返回**: 矩阵乘法结果
- **用途**: 核心矩阵乘法操作

