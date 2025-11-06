# 任务特征
**操作类型**：跨轴broadcast类型，broadcast第一、三根轴；3D Tensor输入，3D Tensor输出
**数据尺寸**：(65536, 128, 16) / (1, 128, 1)，第一、三维需要broadcast，第一维很大，第三维很小
**数据类型**：float32
**任务特点**：操作类型为elementwise，需要对第一维和第三维进行广播，属于跨轴broadcast。

# 关键代码切片

## 优化1：连续broadcast的通用做法

**优化内容**：对于连续broadcast（相邻维度都需要broadcast），通过reshape将这些维度合并为一维，转换为单轴broadcast，简化处理逻辑。

## 优化2：跨轴broadcast的通用做法

**优化内容**：对于跨轴broadcast（不相邻的维度需要broadcast），通用做法是第一维映射到多核上，然后在核内分别对每个维度进行切分。
例如，对于(B, H, W) / (1, H, 1)：
- 选择第一维（如B=65536）映射到多核实现并行
- 在核内对其他维度按需进行切分处理
这种方法适用于大多数跨轴broadcast场景，通过多核并行+核内多维切分，实现并行度和数据粒度的平衡。

## 优化3：最后一维特别小的特殊处理
```python
# ============ 阶段1：多核并行Broadcast（autotune优化）============
input2_broadcast = torch.empty(1, H, W, dtype=input2.dtype, device=input2.device)

# Grid由autotune的NUM_H_CORES决定
grid_broadcast = lambda meta: (meta['NUM_H_CORES'],)
broadcast_kernel_parallel[grid_broadcast](
    input2, input2_broadcast,
    input2.stride(1),
    input2_broadcast.stride(1), input2_broadcast.stride(2),
    H=H, W=W,
)

# ============ 阶段2：Division kernel ============
input1_flat = input1.reshape(B, HW).contiguous()
input2_flat = input2_broadcast.reshape(1, HW).contiguous()
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
**优化内容**：当跨轴broadcast中最后一维特别小时（如W=16），如果直接处理会导致向量化效果差。此时采用**两阶段kernel**策略：
1. **第一阶段kernel（broadcast_kernel）**：先将需要broadcast的维度展开，如将(1, H, 1)先broadcast到(1, H, W)，沿H维度映射到多核并行处理。
2. **第二阶段kernel（elemwise_kernel）**：将3D问题转换为2D处理：
   - 将(B, H, W)和(1, H, W) reshape为2D：(B, HxW)和(1, HxW)
   - 把第一维(B)映射到多核，核内对HW维度切分（SUB_HW=512），向量化维度大大提升

这种方法通过预先broadcast+reshape，避免了最后一维过小导致的向量化效率问题；并且broadcast较小的shape，开销小。

**总结**：[通用优化] 当最后一维很小时，采用两阶段kernel：先broadcast展开+reshape为2D，再进行标准的多核并行处理，提升向量化效率。