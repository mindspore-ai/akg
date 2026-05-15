---
name: triton-cuda-basics
description: "Triton CUDA 编程基础，包括核心概念（program_id、block、grid）、内核函数结构、装饰器用法和标准代码模式。适用于使用 Triton CUDA、需要了解基本语法结构的任意 CUDA 内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
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

## 5. Autotune 用法（仅限静态 shape）

Autotune 通过自动 benchmark 多组配置参数，找到当前硬件和数据规模下的最优配置并缓存，免去手动调参。

### 适用场景

- **推荐使用**：输入 shape 固定或变化范围有限（静态 shape），如固定 batch size 的 MatMul、固定序列长度的 Attention 等
- **禁止使用**：输入 shape 频繁变化（动态 shape）。autotune 根据 `key` 参数缓存最佳 config，动态 shape 下每组新 shape 都会触发一次完整 benchmark，反而严重拖慢性能

### 强制规则

1. **必须写 `restore_value`**：列出 kernel 的**所有输出指针参数名**。autotune benchmark 会对每个 config 反复执行 kernel，`restore_value` 在每次迭代前保存输出张量副本、迭代后恢复原值，防止不同 config 之间的结果互相污染。**不写 `restore_value` 会导致验证失败。**
2. **grid 必须使用 lambda**：`grid = lambda meta: (...)`，确保 grid 能根据当前 config 动态计算。
3. **调用时不传 configs 参数**：autotune 自动传入。
4. **configs 参数必须是 constexpr**：在 kernel 中声明为 `PARAM: tl.constexpr`。
5. **key 参数**：指定哪些输入维度变化时重新 autotune。
6. **num_warps**：控制每个 block 的 warp 数量（常用: 2, 4, 8）。
7. **num_stages**：控制软件流水线级数（常用: 2, 3, 4, 5）。

### 标准写法

```python
# 正确写法：有 restore_value
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
    restore_value=['c_ptr'],  # ⚠ 必须：列出所有输出指针参数名
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pass

grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
matmul_kernel[grid](a, b, c, M, N, K, ...)
```

```python
# 错误：缺少 restore_value → CodeChecker 会拦截，验证会失败
@triton.autotune(
    configs=[...],
    key=[...],
)
@triton.jit
def kernel(input_ptr, output_ptr, ...):
    pass
```

## 最佳实践

1. **始终使用 mask**：处理边界情况，防止越界访问
2. **合理选择 BLOCK_SIZE**：平衡并行度和资源占用（推荐 2 的幂次）
3. **使用 constexpr**：编译时常量，提高性能
4. **注意数据类型**：显式类型转换，避免精度损失
5. **使用 autotune**：自动找到最优配置（包括 num_warps 和 num_stages）
6. **利用 Tensor Core**：MatMul 类算子启用 allow_tf32
