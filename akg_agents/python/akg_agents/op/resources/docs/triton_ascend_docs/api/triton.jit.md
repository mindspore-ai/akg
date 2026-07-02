### @triton.jit
```python
@triton.jit
def kernel_function(...):
    pass
```
- **作用**: 将 Python 函数编译为硬件内核
- **约束**:
  - kernel 内部只能使用 Triton 支持的 Python 子集。
  - 函数内部不要使用 `return`、`break`、`continue` 语句。
  - Python tuple/list 不要作为 runtime 对象在 kernel 内参与 `%`、`//`、比较或广播；stride、padding、dilation 等应拆成标量 meta 参数或 runtime 标量。
  - 需要作为编译期常量的 shape、tile、循环上界应在函数签名中声明为 `tl.constexpr`。
