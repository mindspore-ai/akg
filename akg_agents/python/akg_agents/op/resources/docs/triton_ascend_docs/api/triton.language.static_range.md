### tl.static_range(start, end, step=1)
```python
for k0 in tl.static_range(0, K_TOTAL_PADDED, BLOCK_K):
    ...
```
- **用途**: 编译期展开固定小循环或固定 tile 循环。
- **约束**:
  - `start`、`end`、`step` 必须是编译期可确定值。
  - 对动态真实边界，使用静态 padded 上界循环，并在循环体内用 mask 过滤无效 lane。
  - 展开次数建议 `<=32`，`32~64` 需要谨慎；可能超过 `64` 时改用 `tl.range`、`tl.dot` K tile 或两阶段 reduction。
  - large reduction 不要退化成一输出一 program 的长串行循环；优先让一个 program 覆盖多个输出位置/列，或拆成两阶段 reduction。
