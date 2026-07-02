### tl.constexpr
```python
BLOCK_SIZE: tl.constexpr = 1024
```
- **用途**: 标记编译时常量参数
- **约束**:
  - 必须在 kernel 函数签名中声明。
  - 从 shape 派生的 `K_TOTAL`、`WINDOW`、`BLOCK_POS` 等静态大小应在 host wrapper 中算好，再作为 `tl.constexpr` meta 参数传入。
  - 不要在 `@triton.jit` 内用 runtime 参数构造 `tl.zeros/full/reshape` 的 shape 或 `tl.static_range` 的上界。
