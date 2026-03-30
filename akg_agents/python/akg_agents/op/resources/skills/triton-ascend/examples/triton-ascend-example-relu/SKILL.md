---
name: triton-ascend-example-relu
description: "ReLU 逐元素算子的完整 Triton Ascend 实现示例。展示向量化逐元素操作的标准模式：1D 分块遍历、mask 边界处理、交错循环。当生成 elementwise 类算子时可参考此示例的代码结构。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# ReLU — Triton Ascend 实现示例

```python
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.maximum(x, 0.0)
        tl.store(y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        if not x.is_contiguous():
            x = x.contiguous()
        y = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (self.VEC_CORE_NUM,)
        relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=self.VEC_CORE_NUM)
        return y
```
