---
name: triton-ascend-memory
description: "内存访问优化策略和数据布局技巧"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
---

# 内存访问优化

内存访问是性能的关键瓶颈。本文档提供 Triton Ascend 的内存访问优化策略。

---

## 1. 块大小选择策略

### 调优原则
- **平衡并行度与资源占用**，避免过大或过小
- **BLOCK_SIZE** 常用值：64, 128, 256, 512, 1024
- 过小：并行度不足，硬件利用率低
- 过大：寄存器/共享内存溢出，性能下降

### 推荐设置
- **Element-wise 算子**：BLOCK_SIZE = 1024 或 512
- **Reduce 算子**：BLOCK_SIZE = 256 或 128
- **MatMul 算子**：BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32

---

## 2. 2D 数据内存访问优化

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
    
    B_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    
    # 使用 boundary_check 自动处理边界
    a = tl.load(A_block_ptr, boundary_check=(0, 1))
    b = tl.load(B_block_ptr, boundary_check=(0, 1))
    
    # 计算
    c = tl.dot(a, b)
    
    # 存储结果
    C_block_ptr = tl.make_block_ptr(
        base=C_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(C_block_ptr, c, boundary_check=(0, 1))
```

### Stride 设计要点
- **仔细设计 stride 参数**，错误设置会严重影响性能
- **连续访问**：确保内存访问的连续性和局部性
- **对齐**：Ascend 后端要求 256B 对齐（一般情况），512B 对齐（MatMul 切分）

---

## 3. 连续内存的一维访问优化

### 推荐方案：转连续后用一维访问

张量在内存中连续存储时，可用一维指针遍历，避免多维索引开销。

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

@triton.jit
def elementwise_kernel(
    input_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 一维访问：直接 ptr + offsets
    data = tl.load(input_ptr + offsets, mask=mask)
    result = tl.maximum(data, 0)  # ReLU
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 不推荐方案：使用 stride 访问

```python
# 错误：不推荐：每次 load/store 都需计算 stride 偏移
@triton.jit
def elementwise_kernel_stride(
    input_ptr, output_ptr, 
    M, N, 
    stride_m, stride_n, 
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 每次访问都有额外开销
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 多维索引计算复杂
    offsets = offsets_m[:, None] * stride_m + offsets_n[None, :] * stride_n
    data = tl.load(input_ptr + offsets, mask=...)
    ...
```

### 性能分析

| 方案 | 优势 | 劣势 |
|------|------|------|
| **`.contiguous()` + 一维访问** | 连续内存，缓存友好，访问高效 | 一次性内存拷贝开销 |
| **stride 访问** | 无需拷贝 | 每次 load/store 都需计算偏移，累积开销大 |

**建议**：非连续张量先调用 `.contiguous()` 转换，再用一维访问，整体性能更优。

### 要点总结
- 正确：优先使用 `.contiguous()` 转换 + 一维访问
- 正确：连续内存访问效率远高于 stride 计算开销
- 正确：`torch.empty_like()` 创建的输出默认连续
- 正确：输入输出同形状的 element-wise 算子无需 reshape

---

## 4. 数据布局优化

### 内存连续性
- **确保输入张量连续**：`input.contiguous()`
- **输出张量默认连续**：`torch.empty_like(input)`

### 内存对齐
- **Ascend 256B 对齐**：一般算子（element-wise, reduce）
- **Ascend 512B 对齐**：MatMul 切分（详见 triton-ascend-matmul）

### 缓存友好
- **连续访问**：按内存顺序访问数据
- **局部性**：访问相邻的数据块

---

## 5. 最佳实践

### Element-wise 算子
1. 转连续：`input.contiguous()`
2. 一维访问：`ptr + offsets`
3. BLOCK_SIZE = 1024

### 2D 算子（MatMul、Attention）
1. 使用 `tl.make_block_ptr`
2. 配合 `boundary_check`
3. 注意 stride 设计

### 避免的陷阱
- 错误：非连续张量直接用 stride 访问
- 错误：BLOCK_SIZE 设置过大或过小
- 错误：忘记边界检查导致越界访问
- 错误：未对齐导致性能下降

---

## 6. 调试建议

### 性能问题排查
1. 检查张量是否连续：`tensor.is_contiguous()`
2. 检查 BLOCK_SIZE 是否合理
3. 使用 Profiler 分析内存访问模式

### 常见错误
- **内存访问越界**：检查 mask 和 boundary_check
- **性能不佳**：检查内存连续性和对齐
- **结果错误**：检查 stride 计算是否正确
