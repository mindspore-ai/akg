### tl.load(pointer, mask=None, other=None, boundary_check=None)
```python
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```
- **参数**:
  - `pointer`: 内存指针
  - `mask`: 布尔掩码，True 表示有效位置
  - `other`: 掩码为 False 时的默认值
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **返回**: 加载的张量数据
- **用途**: 从全局内存加载数据
- **约束**:
  - `pointer` 和 `mask` 的 block shape 必须兼容。例如 pointer 为 `(BLOCK_M, BLOCK_K)` 时，mask 也应能广播到 `(BLOCK_M, BLOCK_K)`。
  - 对 padding、tail、反推输入越界等情况，必须用 `mask` 阻止越界地址参与 load；不要先越界 load 再用 `tl.where` 修正数值。
  - `other` 应匹配归约语义：sum/dot 用 `0.0`，min 用 `float("inf")`，max 用 `-float("inf")`。
