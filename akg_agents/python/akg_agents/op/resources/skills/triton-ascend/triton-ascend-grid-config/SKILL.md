---
name: triton-ascend-grid-config
description: "Grid 配置策略和大 shape 算子处理方案"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
---

# Grid 配置策略

Grid 配置是 Triton Kernel 启动的关键。本文档提供 Triton Ascend 的 Grid 配置策略和大 shape 处理方案。

---

## 1. Grid 设置规范

### 维度限制
- **Grid 必须是 tuple 类型**，最多 3 维
- 支持的格式：`(x,)`, `(x, y)`, `(x, y, z)`

```python
# 正确：正确
grid = (100,)
grid = (100, 200)
grid = (100, 200, 50)

# 错误：错误
grid = 100  # 必须是 tuple
grid = [100, 200]  # 必须是 tuple，不能是 list
```

### 大小限制
- **各维度乘积不超过 65535**
- 即 `x * y * z <= 65535`

```python
# 正确：正确
grid = (65535,)  # 65535 <= 65535
grid = (255, 255)  # 255 * 255 = 65025 <= 65535
grid = (40, 40, 40)  # 40 * 40 * 40 = 64000 <= 65535

# 错误：错误
grid = (70000,)  # 70000 > 65535
grid = (300, 300)  # 300 * 300 = 90000 > 65535
```

---

## 2. 切分设置

### Ascend 后端限制
- **BLOCK_SIZE 必须小于 65536**
- **线程块所占内存必须符合硬件限制**

### 多次切分策略
若 shape 过大，单次切分后超过硬件缓存，并且 BLOCK_SIZE 超过限制，可以对循环进行**多次切分**。

```python
@triton.jit
def large_kernel(
    input_ptr, output_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 第一层切分：按 BLOCK_M x BLOCK_N 切分
    for m_start in range(pid_m * BLOCK_M, min((pid_m + 1) * BLOCK_M, M), SUB_BLOCK_M):
        for n_start in range(pid_n * BLOCK_N, min((pid_n + 1) * BLOCK_N, N), SUB_BLOCK_N):
            # 第二层切分：处理更小的块
            offsets_m = m_start + tl.arange(0, SUB_BLOCK_M)
            offsets_n = n_start + tl.arange(0, SUB_BLOCK_N)
            # ... 处理逻辑
```

---

## 3. 大 Shape 算子的 Grid 处理策略

### 问题描述
对于输入 shape 较大的算子，直接按照 `BLOCK_SIZE` 切分得到的 grid 总数可能超过 65535。

**示例**：
- 输入 shape: `(327680, 1024)`
- BLOCK_SIZE: 1024
- Grid 需求: `327680 / 1024 = 320` (每行一个 block)
- 如果有多行：`327680 > 65535` 错误：超限！

---

## 4. 方案 1：交错循环处理（强烈推荐）

### 适用场景
- 按行/按块独立处理的算子
- Element-wise、Reduce、Normalization 等

### 核心思想
- **固定 Grid 为核心数**
- **每个核心以步长方式交错处理数据**
- 负载均衡最好，代码最简洁

### 完整示例

```python
import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def row_processing_kernel(
    input_ptr, 
    output_ptr, 
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 交错处理：每个核心处理 pid, pid+CORE_NUM, pid+2*CORE_NUM, ... 行
    # pid=0 处理第 0, CORE_NUM, 2*CORE_NUM, ... 行
    # pid=1 处理第 1, CORE_NUM+1, 2*CORE_NUM+1, ... 行
    # 这样所有行都会被恰好处理一次，负载均衡
    for row_idx in range(pid, M, CORE_NUM):
        # 计算当前行的指针偏移
        row_ptr = input_ptr + row_idx * stride_m
        out_row_ptr = output_ptr + row_idx * stride_m
        
        # 处理当前行的数据（可根据需要进一步分块）
        for col_start in range(0, N, BLOCK_N):
            col_offsets = col_start + tl.arange(0, BLOCK_N)
            mask = col_offsets < N
            
            data = tl.load(row_ptr + col_offsets * stride_n, mask=mask)
            result = tl.maximum(data, 0)  # 示例：ReLU
            tl.store(out_row_ptr + col_offsets * stride_n, result, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在初始化阶段获取核心数
        try:
            # 向量计算类算子使用 VEC 核心数
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        except:
            self.VEC_CORE_NUM = 48  # Ascend 910B2 默认: 24个AI Core × 2个VEC/Core = 48

    def forward(self, input_tensor):
        M, N = input_tensor.shape
        output_tensor = torch.empty_like(input_tensor)
        
        # 固定 grid 为核心数
        grid = (self.VEC_CORE_NUM,)
        
        row_processing_kernel[grid](
            input_tensor, 
            output_tensor,
            M, N,
            input_tensor.stride(0), 
            input_tensor.stride(1),
            BLOCK_N=256,
            CORE_NUM=self.VEC_CORE_NUM,
        )
        return output_tensor
```

### 负载均衡示例

假设 `M = 1000`, `CORE_NUM = 48`:
- pid=0 处理行: 0, 48, 96, 144, ... (约 21 行)
- pid=1 处理行: 1, 49, 97, 145, ... (约 21 行)
- ...
- pid=47 处理行: 47, 95, 143, ... (约 21 行)

所有核心处理的行数差异不超过 1，负载完美均衡。

---

## 5. 动态获取核心数

### 核心数选择原则

根据算子类型选择对应的核心数：

| 算子类型 | 核心类型 | 默认核心数 (910B2) | 默认核心数 (910B4) |
|---------|---------|-------------------|-------------------|
| **向量计算类** | VEC | 48 (24×2) | 40 (20×2) |
| **矩阵计算类** | CUBE | 24 (24×1) | 20 (20×1) |

**向量计算类算子**：
- Element-wise（add, mul, relu, sigmoid, etc.）
- Reduce（sum, mean, max, min）
- Normalization（softmax, layernorm）

**矩阵计算类算子**：
- MatMul（matmul, bmm, linear）
- Attention（self-attention, cross-attention）

### 完整代码

```python
import torch_npu

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在 __init__ 中获取核心数，只执行一次，避免 forward 中的同步开销
        try:
            # 向量计算类算子使用 VEC 核心数
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
            # 矩阵计算类算子使用 CUBE 核心数
            self.CUBE_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("cube_core_num", 20)
        except:
            self.VEC_CORE_NUM = 40   # Ascend 910B4 默认
            self.CUBE_CORE_NUM = 20  # Ascend 910B4 默认
    
    def forward(self, input_tensor):
        # 根据算子类型选择核心数
        if self.is_vector_op:
            core_num = self.VEC_CORE_NUM
        else:
            core_num = self.CUBE_CORE_NUM
        
        grid = (core_num,)
        kernel[grid](..., CORE_NUM=core_num)
        return output
```

### ⚠️ 重要注意事项

**禁止在 forward 中调用 `get_device_limit`**

```python
# 错误：错误：每次 forward 都调用，触发设备同步
class ModelNew(torch.nn.Module):
    def forward(self, input_tensor):
        core_num = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
        grid = (core_num,)
        ...

# 正确：正确：在 __init__ 中调用，只执行一次
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 48)
    
    def forward(self, input_tensor):
        grid = (self.VEC_CORE_NUM,)
        ...
```

**原因**：`torch_npu` 的 import 和 `get_device_limit` 调用会触发设备同步，在 forward 中频繁调用会严重影响性能。

---

## 6. 方案 2：2D Grid 切分（适用于 2D 算子）

### 适用场景
- 2D 矩阵计算（MatMul, Attention）
- 需要行列双向并行

### 示例

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 每个 block 处理一个 (BLOCK_M, BLOCK_N) 的输出块
    # ...

# Grid 设置
grid_m = triton.cdiv(M, BLOCK_M)
grid_n = triton.cdiv(N, BLOCK_N)

# 检查是否超限
if grid_m * grid_n > 65535:
    # 使用固定核心数 + 交错循环
    grid = (CUBE_CORE_NUM,)
else:
    # 使用 2D Grid
    grid = (grid_m, grid_n)
```

---

## 7. 最佳实践总结

### Element-wise / Reduce 算子
1. 使用固定 VEC 核心数
2. 交错循环处理所有行
3. BLOCK_SIZE = 1024 或 512

### MatMul 算子
1. 优先使用固定 CUBE 核心数
2. 交错循环或 2D Grid（根据 shape 决定）
3. 注意 512B 对齐

### Attention 算子
1. 使用固定 CUBE 核心数
2. Flash Attention 分块策略
3. 在线 Softmax 累加

---

## 8. 常见错误和解决方案

### 错误 1: Grid 超限
```
RuntimeError: Grid size exceeds 65535
```

**解决**：使用固定核心数 + 交错循环

### 错误 2: 性能不佳
```
Kernel 运行缓慢
```

**排查**：
1. 检查核心数是否正确（VEC vs CUBE）
2. 检查是否在 forward 中调用 `get_device_limit`
3. 检查负载是否均衡

### 错误 3: 结果错误
```
输出与预期不符
```

**排查**：
1. 检查交错循环的步长是否正确
2. 检查边界条件和 mask
3. 检查是否所有数据都被处理

---

## 9. 性能调优建议

### Grid 大小调优
1. 小 shape：使用计算出的 grid（如 `triton.cdiv(M, BLOCK_SIZE)`）
2. 大 shape：使用固定核心数

### 负载均衡
1. 交错循环自动均衡
2. 2D Grid 需要注意 M, N 的对齐

### 内存访问
1. 连续访问优先
2. 避免 bank conflict
3. 合理使用 shared memory
