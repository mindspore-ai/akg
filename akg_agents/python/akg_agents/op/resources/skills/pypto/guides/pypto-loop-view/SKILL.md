---
name: pypto-loop-view
description: "pypto.loop + pypto.view 的正确写法：view shape 必须是编译期常量，适用于 matmul、norm、elementwise 等所有 loop 场景"
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matmul,norm,elementwise,reduction,loop,view"
---

# Loop + View：编译期常量规则

## 致命错误：`ValueError: Not concrete value`

**最常见的首次生成错误**。所有 `pypto.view` 的 shape 参数必须是**编译期常量**（字面量或闭包变量）。

```python
# WRONG — min() 含 loop 变量 offset，是运行时值
for idx in pypto.loop(0, num_iters, 1, ...):
    offset = idx * BASIC_BATCH
    current = min(BASIC_BATCH, total_size - offset)      # runtime!
    chunk = pypto.view(x, [current, n], [offset, 0])     # ValueError!
```

**任何含 loop idx 的表达式都是运行时值**，不能用于 view shape。

## 通用正确写法

### 方法 A：确保整除（推荐）

选择 BASIC_BATCH 使 total_size 可整除，或在 forward 中 assert 整除性：

```python
def create_kernel(total_rows, cols, basic_batch):
    assert total_rows % basic_batch == 0
    num_iters = total_rows // basic_batch  # 闭包常量

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((total_rows, cols), ...)) -> ...:
        output = pypto.tensor([total_rows, cols], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)
        for idx in pypto.loop(0, num_iters, 1, name="LOOP", idx_name="idx"):
            off = idx * basic_batch
            chunk = pypto.view(x, [basic_batch, cols], [off, 0])
            result_chunk = chunk * 2.0  # 示例操作
            pypto.assemble(result_chunk, [off, 0], output)
        return output
    return kernel

class ModelNew(torch.nn.Module):
    def forward(self, x):
        total_rows = x.shape[0] * x.shape[1]
        # 调参由 loop_count 空间驱动：先试 16/32，再反推 basic_batch
        target_loop_count = 16
        assert total_rows % target_loop_count == 0
        basic_batch = total_rows // target_loop_count
        ...
```

### 方法 B：主循环 + 尾部

当无法保证整除时：

```python
def create_kernel(total_rows, cols, basic_batch):
    full_iterations = total_rows // basic_batch
    tail = total_rows % basic_batch
    tail_offset = full_iterations * basic_batch
    # full_iterations, tail, tail_offset 都是闭包常量

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((total_rows, cols), ...)) -> ...:
        output = pypto.tensor([total_rows, cols], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)

        for idx in pypto.loop(0, full_iterations, 1, name="LOOP", idx_name="idx"):
            off = idx * basic_batch
            chunk = pypto.view(x, [basic_batch, cols], [off, 0])
            pypto.assemble(chunk * 2.0, [off, 0], output)

        if tail > 0:  # 编译期求值（tail 是闭包常量）
            tail_chunk = pypto.view(x, [tail, cols], [tail_offset, 0])
            pypto.assemble(tail_chunk * 2.0, [tail_offset, 0], output)
        return output
    return kernel
```

## 1D Loop（全局归约/大向量）

```python
LOOP_CHUNKS = 8

def create_frobenius_kernel(flat_size):
    chunk_size = flat_size // LOOP_CHUNKS  # 闭包常量

    @pypto.frontend.jit(...)
    def kernel(x: pypto.Tensor((flat_size,), ...)) -> ...:
        output = pypto.tensor([flat_size], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(16384)
        acc = pypto.zeros([1], dtype=pypto.DT_FP32)
        for i in pypto.loop(0, LOOP_CHUNKS, 1, name="LOOP_ACC", idx_name="i"):
            x_chunk = pypto.view(x, [chunk_size], [i * chunk_size])
            part = pypto.sum(x_chunk * x_chunk, dim=0, keepdim=True)
            acc[:] = acc + part
        norm = pypto.sqrt(acc)
        output[:] = x / norm
        return output
    return kernel

class ModelNew(torch.nn.Module):
    def forward(self, x):
        x_flat = x.reshape(-1)
        assert x_flat.numel() % LOOP_CHUNKS == 0
        ...
```

## 关键原则

1. **view shape 只能用字面量或闭包变量**。loop idx 及含 idx 的表达式是运行时值。
2. **`if tail > 0:` 在编译期求值**（tail 是闭包常量），不是运行时分支。
3. **优先确保整除**，避免尾部处理的复杂度。
4. **Matmul 不要固定 BASIC_BATCH**。先在 `loop_count` 空间选中段（`1~128` 常见从 `16/32` 起步），再反推 `BASIC_BATCH`；必要时扩到 `8/64`，最后再补端点。
