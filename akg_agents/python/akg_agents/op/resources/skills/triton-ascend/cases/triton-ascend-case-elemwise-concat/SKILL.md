---
name: triton-ascend-case-elemwise-concat
description: "Slice+Concat融合算子优化：通过精确切片加载（只load需要部分）和索引计算拼接（避免cat指令）避免中间结果存储和多次内存访问，适用于多输入需要切片后拼接的融合算子场景"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# Slice + Concat 融合算子优化案例

## 任务特征
- **操作类型**：融合算子，6个slice + 1个concat融合在一个kernel中
- **数据尺寸**：7个大小为(128, 50, 128)的输入，切片[128, 32, 48, 48, 48, 48, 48]后在W维度拼接，输出(128, 50, 400)
- **任务特点**：算子融合，避免中间结果的存储和多次内存访问

## 优化 1：精确切片加载

```python
# 只load需要的切片部分，而不是整个输入
# Input 1: 只load前128个元素
w_offs_1 = tl.arange(0, SLICE_1)  # SLICE_1=128
input_offs = base_in_offs + w_offs_1[None, None, :] * stride_in_w
data = tl.load(x1_ptr + input_offs, mask=mask_1, other=0.0)

# Input 2: 只load前32个元素
w_offs_2 = tl.arange(0, SLICE_2)  # SLICE_2=32
input_offs = base_in_offs + w_offs_2[None, None, :] * stride_in_w
data = tl.load(x2_ptr + input_offs, mask=mask_2, other=0.0)
```

### 优化内容
- 在kernel内部只load每个输入需要的切片部分（如128、32、48）
- 通过 `w_offs = tl.arange(0, SLICE_SIZE)` 精确控制load的元素数量
- 减少不必要的内存访问，提高内存带宽利用率

## 优化 2：索引计算实现拼接

```python
# 通过调整输出索引实现拼接，而非使用triton的cat指令
w_out_offset = 0

# Input 1写入位置: output[0:128]
output_offs = base_out_offs + (w_out_offset + w_offs_1)[None, None, :] * stride_out_w
tl.store(output_ptr + output_offs, data, mask=mask_1)
w_out_offset += SLICE_1  # 更新为128

# Input 2写入位置: output[128:160]
output_offs = base_out_offs + (w_out_offset + w_offs_2)[None, None, :] * stride_out_w
tl.store(output_ptr + output_offs, data, mask=mask_2)
w_out_offset += SLICE_2  # 更新为160
```

### 优化内容
- 通过维护输出偏移量（w_out_offset）并动态调整输出地址索引，将不同输入的数据写入到输出的不同位置
- 避免使用triton的cat指令，直接通过地址计算完成拼接
- 减少中间步骤和额外的数据搬运开销

### 总结
1. 对于concat操作，应在kernel内精确load需要的切片部分，避免load完整数据后再切片
2. 可通过索引计算直接将数据store到目标位置实现拼接，无需使用额外的cat指令
3. 算子融合可以避免中间结果的存储和多次内存访问，提升整体性能
