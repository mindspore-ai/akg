### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **作用**: 将 Python 函数编译为硬件内核
- **约束**: 函数内部不能使用 `return`、`break`、`continue` 语句

