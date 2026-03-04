---
name: triton-ascend-case-elemwise-broadcast-3d
description: "跨轴3D广播优化（最后一维很小）：采用两阶段kernel策略（先broadcast展开+reshape为2D，再标准多核处理）提升向量化效率，适用于跨轴broadcast且最后一维特别小（<20）导致向量化效果差的场景"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  hardware: "Atlas A2, Atlas A3"
---

# 跨轴 3D Broadcast 优化案例

## 任务特征
- **操作类型**：跨轴broadcast，broadcast第一、三根轴
- **数据尺寸**：(65536, 128, 16) / (1, 128, 1)
- **特点**：第一维很大，第三维很小，属于跨轴broadcast

## 优化：两阶段 Kernel 策略

当跨轴broadcast中最后一维特别小时（如W=16），如果直接处理会导致向量化效果差。

### 阶段1：Broadcast Kernel（多核并行）

```python
# 先将(1, H, 1) broadcast到(1, H, W)
input2_broadcast = torch.empty(1, H, W, dtype=input2.dtype, device=input2.device)

grid_broadcast = lambda meta: (meta['NUM_H_CORES'],)
broadcast_kernel_parallel[grid_broadcast](
    input2, input2_broadcast,
    input2.stride(1),
    input2_broadcast.stride(1), input2_broadcast.stride(2),
    H=H, W=W,
)
```

### 阶段2：Division Kernel（Reshape为2D）

```python
# 将3D问题转换为2D处理
input1_flat = input1.reshape(B, HW).contiguous()  # (B, HxW)
input2_flat = input2_broadcast.reshape(1, HW).contiguous()  # (1, HxW)
output_flat = torch.empty(B, HW, dtype=input1.dtype, device=input1.device)

grid_div = lambda meta: (meta['NUM_CORES'],)
div_flatten_kernel[grid_div](
    input1_flat, input2_flat, output_flat,
    B, HW,
    input1_flat.stride(0), input1_flat.stride(1),
    input2_flat.stride(1),
    output_flat.stride(0), output_flat.stride(1),
)
```

### 优化内容
1. **第一阶段kernel**：先将需要broadcast的维度展开，沿H维度映射到多核并行处理
2. **第二阶段kernel**：将3D reshape为2D，把第一维(B)映射到多核，核内对HW维度切分（SUB_HW=512），向量化维度大大提升

这种方法通过预先broadcast+reshape，避免了最后一维过小导致的向量化效率问题。

## 通用优化方案

### 连续broadcast（相邻维度）
通过reshape将相邻维度合并为一维，转换为单轴broadcast。

### 跨轴broadcast（不相邻维度）
- 第一维映射到多核上实现并行
- 核内对其他维度按需进行切分

### 总结
当跨轴broadcast中最后一维特别小时，采用两阶段kernel：先broadcast展开+reshape为2D，再进行标准的多核并行处理，提升向量化效率。
