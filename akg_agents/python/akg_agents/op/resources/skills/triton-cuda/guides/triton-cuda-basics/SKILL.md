---
name: triton-cuda-basics
description: "Triton CUDA 编程基础，包括核心概念（program_id、block、grid）、内核函数结构、装饰器用法和标准代码模式。适用于使用 Triton CUDA、需要了解基本语法结构的任意 CUDA 内核代码生成场景"
level: L4
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton-cuda
  operator_patterns: "all"
---

# Triton CUDA 编程基础

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `@triton.jit` 装饰的 Python 函数，编译后在 GPU 上并行执行
- **特点**: 每个内核实例处理数据的一个子集，通过程序 ID 区分

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置，如 `(num_blocks_x, num_blocks_y)`
- **块**: 每个程序实例处理的数据块大小，如 `BLOCK_SIZE = 1024`
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存 (Global Memory)**: 主内存（HBM），所有程序可访问，延迟高，带宽大
- **共享内存 (Shared Memory)**: SM 内共享，延迟低，容量有限（通常 48-164 KB/SM）
- **寄存器 (Registers)**: 每个线程私有，最快访问

### CUDA GPU 架构要点
- **SM (Streaming Multiprocessor)**: GPU 基本计算单元
- **Warp**: 32 个线程为一组并行执行
- **Tensor Core**: 专用矩阵计算单元（Ampere/Hopper 架构）

## 2. 标准内核结构（五步模式）

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

## 3. 内核启动方式

### 函数形式
```python
def launch_kernel(input_tensor, output_tensor):
    BLOCK_SIZE = 1024  
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
    
    kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

### ModelNew 类格式（推荐）
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

## 4. 边界处理

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

## 5. Autotune 使用教程

Autotune 是 Triton 的自动性能优化机制，通过尝试不同的配置参数组合，自动找到最优的内核执行配置。

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # 当这些参数变化时触发重新autotune
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, # configs中的参数必须声明为constexpr
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
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
5. **num_warps**: CUDA 特有，控制每个 block 的 warp 数量（常用: 2, 4, 8）
6. **num_stages**: 控制软件流水线级数（常用: 2, 3, 4, 5）

## 最佳实践

1. **始终使用 mask**：处理边界情况，防止越界访问
2. **合理选择 BLOCK_SIZE**：平衡并行度和资源占用（推荐 2 的幂次）
3. **使用 constexpr**：编译时常量，提高性能
4. **注意数据类型**：显式类型转换，避免精度损失
5. **使用 autotune**：自动找到最优配置（包括 num_warps 和 num_stages）
6. **利用 Tensor Core**：MatMul 类算子启用 allow_tf32
