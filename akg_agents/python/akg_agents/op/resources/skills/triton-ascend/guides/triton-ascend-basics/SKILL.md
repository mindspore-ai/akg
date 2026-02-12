---
name: triton-ascend-basics
description: "Triton Ascend 编程基础，包括核心概念（program_id、block、grid）、内核函数结构、装饰器用法和标准代码模式。适用于初次使用 Triton Ascend、需要了解基本语法结构的任意内核代码生成场景"
level: L3
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
  operator_patterns: "all"
---

# Triton Ascend 编程基础

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `@triton.jit` 装饰的 Python 函数，编译后在硬件加速器上并行执行
- **特点**: 每个内核实例处理数据的一个子集，通过程序 ID 区分

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置，如 `(num_blocks_x, num_blocks_y)`
- **块**: 每个程序实例处理的数据块大小，如 `BLOCK_SIZE = 1024`
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存**: 主内存，所有程序可访问，延迟高
- **共享内存**: 块内共享，延迟低，容量有限
- **寄存器**: 每个线程私有，最快访问

## 2. 标准内核结构

所有 Triton 内核都遵循相同的五步结构模式：

```python
@triton.jit
def standard_kernel(
    output_ptr, input_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取程序 ID 和计算偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 2. 创建边界掩码
    mask = offsets < n_elements
    
    # 3. 加载数据
    data = tl.load(input_ptr + offsets, mask=mask)
    
    # 4. 执行计算
    result = compute_function(data)
    
    # 5. 存储结果
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 内核启动方式

```python
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        output_tensor = torch.empty_like(input_tensor)
        BLOCK_SIZE = 1024  
        grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
        
        kernel[grid](
            output_tensor, input_tensor, input_tensor.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return output_tensor
```

## 3. 边界处理

### 使用 mask 处理边界
```python
# 基本边界检查
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
data = tl.load(ptr + offsets, mask=mask, other=0.0)
```

### 条件计算
```python
# 使用 tl.where 进行条件选择
result = tl.where(condition, true_value, false_value)

# 复杂条件的掩码组合
valid_mask = (offsets < n_elements) & (offsets >= 0)
data = tl.load(ptr + offsets, mask=valid_mask, other=0.0)
```

## 4. Autotune 使用教程

Autotune 是 Triton 的自动性能优化机制，通过尝试不同的配置参数组合，自动找到最优的内核执行配置。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}),
    ],
    key=['M', 'N', 'K'],  # 当这些参数变化时触发重新autotune
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, # configs中的参数必须声明为constexpr
    BLOCK_SIZE_N: tl.constexpr, # configs中的参数必须声明为constexpr
    BLOCK_SIZE_K: tl.constexpr, # configs中的参数必须声明为constexpr
):
    # kernel实现
    pass

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
        
        # 关键：调用时不要传递configs中的参数（BLOCK_SIZE_M等）
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            # 不要写：BLOCK_SIZE_M=128  # 错误！autotune会自动传入
        )
        return c
```

### Autotune 关键要点

1. **grid必须使用lambda**：`grid = lambda meta: (...)`
2. **不要传递configs参数**：调用kernel时不要传`BLOCK_SIZE`等autotune的参数
3. **configs参数必须是constexpr**：在kernel中声明为`PARAM: tl.constexpr`
4. **key参数**：指定哪些输入维度变化时重新autotune

**注意**：不要对 'num_warps', 'num_ctas', 'num_stages', 'num_buffers_warp_spec', 'num_consumer_groups', 'reg_dec_producer', 'reg_inc_consumer', 'maxnreg' 进行修改调优，当前 Ascend 后端不支持。