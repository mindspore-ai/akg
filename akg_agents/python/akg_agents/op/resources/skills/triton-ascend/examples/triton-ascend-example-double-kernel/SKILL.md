---
name: triton-ascend-example-double-kernel
description: "双内核调用模式的 Triton Ascend 实现示例。展示在 forward 中先后调用两个 kernel 的标准写法：中间结果缓冲区分配、两次 kernel 启动。适用于需要分阶段计算的融合算子。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  framework: torch
---

# Double Kernel（双内核调用）— Triton Ascend 示例

当一个算子需要分两步计算（如先做变换再做归约），可在 `forward` 中依次启动两个 kernel：

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        intermediate = torch.empty_like(x)
        output = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        grid = (self.VEC_CORE_NUM,)

        kernel_stage1[grid](x, intermediate, ..., CORE_NUM=self.VEC_CORE_NUM)
        kernel_stage2[grid](intermediate, output, ..., CORE_NUM=self.VEC_CORE_NUM)

        return output
```

**要点**：
- 中间缓冲区用 `torch.empty_like` 或指定 shape 分配
- 确保 stage1 写完后 stage2 再读（Triton Ascend 默认隐式同步）
