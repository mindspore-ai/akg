---
name: pypto-case-loss-crossentropy
description: "模式 D 示例：Loss — CrossEntropyLoss，展示多输入 kernel、两段 tile、softmax+gather+sum、标量输出"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "loss,reduction,gather,softmax"
---

# 模式 D：Loss — CrossEntropyLoss

```python
def create_cross_entropy_kernel(batch, num_classes):
    @pypto.frontend.jit(runtime_options=..., debug_options=...)
    def kernel(
        predictions: pypto.Tensor((batch, num_classes), pypto.DT_FP32),
        targets: pypto.Tensor((batch,), pypto.DT_INT64),
    ) -> pypto.Tensor((1,), pypto.DT_FP32):
        output = pypto.tensor([1], pypto.DT_FP32)
        # Phase 1: per-sample softmax + gather
        pypto.set_vec_tile_shapes(1024, 16)
        log_probs = pypto.log(pypto.softmax(predictions, dim=1))
        targets_i32 = pypto.cast(targets, pypto.DT_INT32)
        idx = pypto.unsqueeze(targets_i32, 1)
        picked = pypto.gather(log_probs, dim=1, index=idx)
        neg_picked = pypto.mul(picked, -1.0)
        # Phase 2: batch reduction
        pypto.set_vec_tile_shapes(2048, 8)
        total = pypto.sum(neg_picked, dim=0, keepdim=False)
        output[:] = total / batch
        return output
    return kernel
```

forward：assert → contiguous → 调 kernel → `reshape(1,)`

## 模式要点
- **两段 tile**：不同计算阶段用不同 tile 配置
- `pypto.cast(targets, DT_INT32)` — INT64 输入需转 INT32
- `pypto.unsqueeze` + `pypto.gather` — 按 index 取元素
- `pypto.mul(x, -1.0)` — 取反的标准写法（规则 R2 的应用）
- 标量输出：`pypto.tensor([1], ...)` + `output[:] = scalar`
- 逐元素 loss（MSE/Huber 等）更简单：所有输入 `reshape(-1)` 用 1D kernel
