### tl.reshape(input, shape)
```python
y = tl.reshape(x, (BLOCK_M, BLOCK_N))
```
- **参数**:
  - `input`: 输入 block tensor
  - `shape`: 目标形状
- **返回**: reshape 后的 block tensor
- **约束**:
  - `shape` 必须是编译期可确定 tuple，元素总数需与输入一致。
  - 不要用 `tl.reshape` 处理 host 侧 tensor 布局转换；host tensor 的 `.reshape/.view/.contiguous` 应在 `forward()` 中完成。
