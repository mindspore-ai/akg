---
name: triton-ascend-grid-config
description: "Grid/Block 配置策略，包括核数选择、并行度调优、二次切分和大 shape 算子处理方案。适用于需要确定 kernel 启动参数、优化多核并行效率、或处理超大规模数据的内核代码生成场景"
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

### 大小限制
- **各维度乘积不超过 65535**
- 即 `x * y * z <= 65535`

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

### 方案 1：交错循环处理（强烈推荐）

#### 适用场景
- 按行/按块独立处理的算子
- Element-wise、Reduce、Normalization 等

#### 核心思想
- **固定 Grid 为核心数**
- **每个核心以步长方式交错处理数据**
- 负载均衡最好，代码最简洁

#### 完整示例

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

#### 优点
- 代码极其简洁，一行for循环解决问题
- 负载天然均衡（每个核心处理的任务数差最多为1）
- 无需计算ELEMENTS_PER_GRID等复杂参数
- 适用于任意大小的输入shape

---

#### 动态获取核心数

根据算子类型选择对应的核心数，**必须在`__init__`中获取**（避免forward中重复调用导致同步开销）：
- **向量计算类算子**（element-wise、softmax、归一化等）：使用VEC核心数
- **矩阵计算类算子**（matmul、attention等）：使用CUBE核心数

#### 完整代码

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

#### 重要注意事项

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

### 方案 2：连续分块处理

#### 适用场景
- 连续内存访问优化

#### 示例

```python
```python
import torch_npu

# 示例：处理大shape的向量操作
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,  # 每个grid负责的元素总数
):
    pid = tl.program_id(0)
    
    # 计算当前grid负责的数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)
    
    # 分块处理当前grid负责的数据
    for block_start in range(grid_start, grid_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < grid_end
        
        # 加载、计算、存储
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute_function(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 启动方式
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 在__init__中获取核心数
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, input_tensor):
        n_elements = input_tensor.numel()
        BLOCK_SIZE = *  # 设置为尽可能大的合适的每次处理的值，充分利用ub
        MAX_GRID_SIZE = self.VEC_CORE_NUM  # 使用初始化时获取的核心数
        
        # 计算每个grid需要处理的元素数
        ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
        # 向上取整到BLOCK_SIZE的倍数，确保循环能完整处理
        ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE
        
        # 计算实际需要的grid数量
        grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)
        
        output_tensor = torch.empty_like(input_tensor)
        large_vector_kernel[grid_size,](
            input_tensor, output_tensor, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
        )
        return output_tensor
```