---
name: triton-ascend-case-index-histogram
description: "直方图统计（histogram）优化：预排序+二分查找降低算法复杂度（O(n×m)→O(n log n + m log n)，性能提升19倍），转换为float32调用Vec Core硬件加速排序，适用于大规模统计类操作（50万+元素）"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Histogram 直方图统计优化案例

## 任务特征
- **操作类型**：直方图统计，统计每个专家ID出现的次数
- **数据尺寸**：输入索引(65536, 8)，专家数量365
- **特点**：需要优化算法复杂度，从O(n×m)降至O(n log n + m log n)

## 优化 1：预排序 + 二分查找

### 错误：简单方式：遍历统计 O(n×m)

```python
count = 0
for i in range(total_elements):  # 524288次迭代
    val = tl.load(indices_ptr + i)
    if val == expert_idx:
        count += 1
```

**问题**：复杂度O(n×m) = 524288 × 365 ≈ 1.9亿次操作

### 正确：优化方式：预排序+二分查找 O(n log n + m log n)

```python
# 预排序：O(n log n)
indices_flat = indices.flatten().to(torch.float32)
sorted_indices, _ = torch.sort(indices_flat)

# Triton kernel内二分查找：每个expert执行O(log n)
@triton.jit
def histogram_kernel(sorted_indices_ptr, splits_ptr, total_elements):
    expert_idx = tl.program_id(0)
    expert_id = expert_idx.to(tl.float32)
    
    # 二分查找下界（O(log n)，约19次迭代）
    left, right = 0, total_elements - 1
    start_pos = total_elements
    while left <= right:
        mid = (left + right) // 2
        mid_val = tl.load(sorted_indices_ptr + mid)
        if mid_val < expert_id:
            left = mid + 1
        else:
            if mid_val == expert_id:
                start_pos = tl.minimum(start_pos, mid)
            right = mid - 1
    
    # 二分查找上界（类似逻辑）
    # ...
    count = end_pos - start_pos + 1
```

**性能对比**：
- 遍历统计：1.9亿次操作
- 预排序+二分查找：约1000万次操作
- **性能提升：约19倍**

## 优化 2：Float32 类型转换（Vec Core加速）

### 错误：简单方式：直接使用 int32

```python
indices_flat = indices.flatten()  # int32
sorted_indices, _ = torch.sort(indices_flat)  # 可能调用AI CPU
```

**问题**：可能回退到AI CPU排序，性能较差

### 正确：优化方式：转换为 float32

```python
indices_flat = indices.flatten().to(torch.float32)  # 转换为float32
sorted_indices, _ = torch.sort(indices_flat)  # 调用Vec Core排序
```

### 优化内容
- Ascend芯片包含AI Core、Vec Core、AI CPU
- Vec Core对float32类型的排序操作有专门优化，支持SIMD并行
- int32排序可能回退到AI CPU，性能较差
- 索引值范围远小于float32精度范围（2^23），转换不会损失精度

### 总结
1. **[算法优化]** 对于统计类操作，应优先考虑预排序+二分查找，将O(n×m)复杂度降至O(n log n + m log n)
2. **[底层接口优化]** 在Ascend平台上，对于大规模排序，应使用float32类型调用Vec Core硬件加速
3. 365个专家的二分查找可以并行执行，每个线程块独立处理一个专家
