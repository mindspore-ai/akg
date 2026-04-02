---
name: triton-ascend-elementwise
description: "适用于纯逐元素(element-wise)类算子的优化指南。当算子的核心计算是对张量每个元素独立执行相同操作、无跨元素依赖时应选择此指南，典型算子包括：relu, sigmoid, tanh, gelu, selu, leaky_relu, elu, swish, softplus, hardsigmoid, hardtanh, softsign, exp, log, sqrt, pow, add, mul, sub, div, abs, neg, clamp, cast(类型转换), where, fill, copy 等。也适用于涉及标量广播(broadcast)的运算。不适用于需要跨元素归约(如 sum/mean/max)或矩阵乘法的算子。如果算子同时包含逐元素计算和全局归约（如损失函数 MSELoss、HuberLoss、HingeLoss），应选择 elementwise-reduce-fused 指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise"
---

# Element-wise 算子编写指南

## 编写模式

Element-wise 算子的核心特征：每个输出元素仅依赖对应位置的输入元素，无跨元素依赖。
通用写法是将张量展平为 1D，用交错循环按 block 遍历全部元素。

### 标准写法

```python
@triton.jit
def elementwise_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        y = compute(x)  # 替换为具体计算
        tl.store(output_ptr + offsets, y, mask=mask)

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
        n = x.numel()
        grid = (self.VEC_CORE_NUM,)
        elementwise_kernel[grid](x, y, n, BLOCK_SIZE=1024, CORE_NUM=self.VEC_CORE_NUM)
        return y
```

**要点**：
- `.contiguous()` 保证一维指针连续访问，避免 stride 计算
- `torch.empty_like` 创建输出（不用 zeros，省初始化开销）
- `forward` 的参数签名和数量必须与原始 `Model.forward` 一致

## 优化技巧

### 1. 连续内存访问

展平为一维后用连续偏移访问，缓存命中率最高：
- 非连续张量先 `.contiguous()`
- 用 `x.numel()` 获取总元素数，忽略原始 shape

### 2. BLOCK_SIZE 选择

- 推荐 1024-2048，平衡流水效率和 UB 占用
- 数据量很小时可降到 256-512
- 数据量很大时不需要增大 BLOCK_SIZE，交错循环自动均衡

### 3. 数值稳定性

- `exp` 前减最大值防溢出
- `sqrt` 前确保非负：`tl.maximum(x, 0.0)` 或 `tl.maximum(x, eps)`
- 中间计算用 float32 累加，最后转回目标精度

### 4. 融合多步计算

连续的 elementwise 操作应融合在同一个 kernel 内，避免多次 GM 读写：

```python
# 融合 x -> relu -> scale -> add_bias
y = tl.maximum(x, 0.0)  # relu
y = y * scale            # scale
y = y + bias             # add_bias
```

### 5. 广播处理

当一个输入是标量或需要广播时，在 kernel 外部处理或在 kernel 中用常量加载：

```python
# 标量作为 kernel 参数传入
@triton.jit
def scale_kernel(x_ptr, out_ptr, scale_val, n, BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr):
    ...
    y = tl.load(x_ptr + offs, mask=mask, other=0.0) * scale_val
```
