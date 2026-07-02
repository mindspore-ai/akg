### tl.store(pointer, value, mask=None, boundary_check=None)
```python
tl.store(ptr + offsets, result, mask=mask)
```
- **参数**:
  - `pointer`: 内存指针
  - `value`: 要存储的值
  - `mask`: 布尔掩码，True 表示有效位置
  - `boundary_check`: 边界检查维度 (0, 1) 或 None
- **用途**: 将数据存储到全局内存
- **约束**:
  - `pointer`、`value`、`mask` 的 block shape 必须兼容。
  - store mask 必须覆盖所有输出 tail 维度，避免多个 program 写同一个输出元素。
