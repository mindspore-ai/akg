---
name: triton-ascend-reduce
description: "适用于归约(reduce)类算子的优化指南。当算子需要沿一个或多个维度对数据进行聚合计算时应选择此指南，典型算子包括：sum, mean, max, min, prod, argmax, argmin, cumsum, cumprod, softmax, logsoftmax, layernorm, rmsnorm, groupnorm, instancenorm, batchnorm, l1norm, l2norm, var, std 等。也适用于含归约子步骤的复合算子(如 normalize 类)。不适用于纯逐元素运算或矩阵乘法，但与 attention 机制中的 softmax 部分有重叠——若算子核心是注意力计算，应优先选择 attention 指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "reduce"
---

# Reduce 算子优化

> 适用于需要聚合多个值的归约操作

## 适用算子

**基础归约**: sum, mean, max, min, prod
**归一化**: softmax, logsoftmax, layernorm, batchnorm
**统计**: variance, std

## 通用归约策略

### 1. 块内归约 + 原子操作

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作写回全局内存
    tl.atomic_add(output_ptr, block_sum)
```

### 2. 数值稳定性处理

**关键**: 对于涉及 exp 的操作（softmax、logsoftmax），必须减去最大值防止溢出。

```python
# 错误：错误：直接 exp 可能溢出
scores = tl.math.exp2(x)

# 正确：正确：减去最大值
max_val = tl.max(x, axis=0)
scores = tl.math.exp2(x - max_val)
```