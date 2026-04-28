### tl.cdiv(a, b)
```python
result = tl.cdiv(offset, BLOCK_SIZE)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果 ⌈a/b⌉
- **用途**: kernel内部使用，计算向上整除结果，等价于 `(a + b - 1) // b`

