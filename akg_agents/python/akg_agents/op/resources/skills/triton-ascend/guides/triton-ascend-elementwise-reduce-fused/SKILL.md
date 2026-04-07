---
name: triton-ascend-elementwise-reduce-fused
description: "适用于同时包含逐元素计算和全局归约两个阶段的复合算子。典型算子包括：损失函数（MSELoss, HuberLoss, HingeLoss, SmoothL1Loss, CrossEntropyLoss, KLDivLoss, CosineSimilarityLoss, TripletMarginLoss 等）、以及自定义的先逐元素变换再全局聚合的算子。这类算子的计算模式是：第一步对张量每个元素独立执行变换（差值、平方、clamp 等），第二步对变换结果做全局或按维度归约（sum/mean）得到标量或低维结果。与纯 elementwise 或纯 reduce 不同，这类算子需要在同一个 kernel 中融合两个阶段以避免中间结果的额外 GM 读写。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "elementwise_reduce_fused"
---

# Elementwise + Reduce 融合算子指南

> 适用于先逐元素计算、再全局归约的复合算子（损失函数等）

## 计算模式

这类算子的通用流程：
1. **Elementwise 阶段**：对输入张量逐元素执行变换（如差值、平方、clamp、log 等）
2. **Reduce 阶段**：对变换结果做全局归约（sum / mean），得到标量或低维输出

## 融合 Kernel 写法

将 elementwise 计算和局部归约放在同一个 kernel 中，避免中间结果写回 GM：

```python
@triton.jit
def fused_loss_kernel(
    pred_ptr, target_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    local_sum = tl.zeros((1,), dtype=tl.float32)

    for block_id in range(pid, num_blocks, CORE_NUM):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        # Elementwise 阶段
        diff = pred - target
        loss_elem = diff * diff  # MSELoss 为例

        # 块内归约
        local_sum += tl.sum(loss_elem, axis=0)

    # 跨块归约
    tl.atomic_add(output_ptr, local_sum / n_elements)
```

## 关键要点

1. **单 kernel 融合**：elementwise 变换和归约在同一 kernel 完成，中间结果仅存在于寄存器/UB 中
2. **原子操作汇总**：多个 program 的局部结果通过 `tl.atomic_add` 汇聚到全局输出
3. **reduction 参数**：注意 PyTorch 损失函数的 `reduction` 参数（`'mean'`/`'sum'`/`'none'`），`'none'` 时退化为纯 elementwise
4. **使用 VEC_CORE_NUM**：此类算子不涉及 `tl.dot`，使用向量核心
5. **数值稳定性**：中间计算用 float32，避免半精度溢出
