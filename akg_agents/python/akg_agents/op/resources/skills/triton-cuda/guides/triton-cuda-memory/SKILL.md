---
name: triton-cuda-memory
description: "CUDA GPU 内存访问优化策略，包括共享内存利用、合并访存、Bank Conflict 避免和数据布局优化技巧。适用于内存带宽受限、需要优化全局内存访问效率、或处理大规模数据的 CUDA 内核性能优化场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# 内存访问优化

内存访问是 GPU 性能的关键瓶颈。本文档提供 Triton CUDA 的内存访问优化策略。

---

## 1. GPU 内存层次

### 内存带宽和延迟

| 内存类型 | 带宽 (A100) | 延迟 | 容量 |
|---------|-------------|------|------|
| 寄存器 | ~19 TB/s | 1 cycle | 256 KB/SM |
| 共享内存 | ~19 TB/s | ~20 cycles | 164 KB/SM |
| L2 缓存 | ~5 TB/s | ~100 cycles | 40 MB |
| 全局内存 (HBM) | ~2 TB/s | ~400 cycles | 40/80 GB |

### 优化原则

- **减少全局内存访问**: 利用共享内存和寄存器
- **合并访问 (Coalesced Access)**: 同一 warp 内线程访问连续地址
- **提高 L2 缓存命中率**: 通过 Grouped Ordering 等技术

---

## 2. 合并访问 (Coalesced Access)

### 什么是合并访问？

当同一 warp 中的 32 个线程访问连续的内存地址时，GPU 可以将这些请求合并为一次或少量内存事务，大幅提高带宽利用率。

```python
# 正确：合并访问（连续地址）
@triton.jit
def coalesced_kernel(input_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 连续偏移
    mask = offsets < n
    data = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, data, mask=mask)

# 错误：非合并访问（跳跃地址）
@triton.jit
def strided_kernel(input_ptr, output_ptr, n, stride, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # 每个线程跳跃 stride 个元素，导致非合并访问
    strided_offsets = offsets * stride
    mask = strided_offsets < n
    data = tl.load(input_ptr + strided_offsets, mask=mask)
```

---

## 3. 块大小选择策略

### 调优原则
- **平衡并行度与资源占用**，避免过大或过小
- **BLOCK_SIZE** 常用值：128, 256, 512, 1024
- 过小：并行度不足，无法充分利用 warp
- 过大：寄存器/共享内存溢出，occupancy 下降

### 推荐设置
- **Element-wise 算子**：BLOCK_SIZE = 1024 或 512
- **Reduce 算子**：BLOCK_SIZE = triton.next_power_of_2(n_cols)
- **MatMul 算子**：BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32-64

---

## 4. 2D 数据内存访问优化

### 优先使用 `tl.make_block_ptr`

对于 2D 数据（如矩阵），**优先使用 `tl.make_block_ptr` 配合 `boundary_check`**，可自动优化内存合并。

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 创建 2D Block Pointer
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),  # Row-major
    )
    
    # 使用 boundary_check 自动处理边界
    a = tl.load(A_block_ptr, boundary_check=(0, 1))
```

### Stride 设计要点
- **仔细设计 stride 参数**，错误设置会严重影响性能
- **连续访问**：确保内存访问的连续性和局部性
- **行主序 (Row-major)**: PyTorch 默认，stride(0) > stride(1)

---

## 5. 连续内存的一维访问优化

### 推荐方案：转连续后用一维访问

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # 非连续张量转为连续（一次性开销）
        if not input_tensor.is_contiguous():
            input_tensor = input_tensor.contiguous()
        
        output_tensor = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        elementwise_kernel[grid](
            input_tensor, output_tensor, 
            n_elements, 
            BLOCK_SIZE=1024
        )
        return output_tensor
```

### 性能对比

| 方案 | 优势 | 劣势 |
|------|------|------|
| **`.contiguous()` + 一维访问** | 合并访问，缓存友好 | 一次性内存拷贝开销 |
| **stride 访问** | 无需拷贝 | 非合并访问，累积开销大 |

**建议**：非连续张量先调用 `.contiguous()` 转换，再用一维访问，整体性能更优。

---

## 6. L2 缓存优化

### Grouped Ordering

对于 MatMul 等 2D 算子，通过分组遍历提高 L2 缓存命中率：

```python
# 标准遍历：L2 缓存利用率低
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n

# Grouped Ordering：L2 缓存利用率高
GROUP_SIZE_M = 8
num_pid_in_group = GROUP_SIZE_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

### swizzle2d

```python
task_m, task_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE)
```

---

## 7. 软件流水线 (Software Pipelining)

### num_stages 参数

通过 `num_stages` 控制预取级数，隐藏内存延迟：

- **num_stages=2**: 最少的共享内存使用
- **num_stages=3-4**: 通常最优
- **num_stages=5+**: 可能超出共享内存限制

```python
@triton.autotune(
    configs=[
        triton.Config({...}, num_stages=2, num_warps=4),
        triton.Config({...}, num_stages=3, num_warps=4),
        triton.Config({...}, num_stages=4, num_warps=8),
    ],
    key=[...],
    restore_value=['output_ptr'],  # 必须：列出所有输出指针参数名
)
```

---

## 8. 最佳实践

### Element-wise 算子
1. 转连续：`input.contiguous()`
2. 一维访问：`ptr + offsets`
3. BLOCK_SIZE = 1024

### 2D 算子（MatMul、Attention）
1. 使用 `tl.make_block_ptr`
2. 配合 `boundary_check`
3. Grouped Ordering 优化 L2 缓存
4. 合理设置 num_stages

### 避免的陷阱
- 非连续张量直接用 stride 访问
- BLOCK_SIZE 设置过大导致 occupancy 下降
- 忘记边界检查导致越界访问
- 忽略 L2 缓存优化

---

## 9. 调试建议

### 性能问题排查
1. 检查张量是否连续：`tensor.is_contiguous()`
2. 检查内存访问是否合并
3. 使用 Nsight Compute 分析内存带宽利用率
4. 检查 occupancy 是否合理

### 常见错误
- **内存访问越界**：检查 mask 和 boundary_check
- **性能不佳**：检查合并访问和 L2 缓存优化
- **结果错误**：检查 stride 计算是否正确
