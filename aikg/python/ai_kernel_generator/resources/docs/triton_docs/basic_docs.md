# Triton 编程基础教程

本文档介绍 Triton 的核心概念和标准编程模式，通过详细示例帮助理解如何构建内核。

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
def launch_kernel(input_tensor, output_tensor):
    BLOCK_SIZE = 1024  
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
    
    kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(),
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

## 3. 三大编程模式

### 3.1 向量操作模式
适用于元素级运算：加法、乘法、激活函数等。

```python
@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    
    tl.store(c_ptr + offsets, c, mask=mask)
```

### 3.2 归约模式
适用于求和、最大值、最小值等聚合操作。

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 加载数据
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作写回全局内存
    if pid == 0:  # 只有第一个块写入结果
        tl.atomic_add(output_ptr, block_sum)
```

### 3.3 矩阵乘法模式
使用块指针高效处理 2D 数据。

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # 获取程序 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 主循环：沿 K 轴迭代
    for k in range(0, K, BLOCK_SIZE_K):
        # 创建块指针
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, k), 
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
            offsets=(k, pid_n * BLOCK_SIZE_N), 
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
        )
        
        # 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # 矩阵乘加
        accumulator += tl.dot(a, b)
    
    # 存储结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0)
    )
    tl.store(c_block_ptr, accumulator, boundary_check=(0, 1))
```

## 4. 边界处理示例

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

## 5. auto-tune

Triton 提供了 `@triton.autotune` 装饰器，可以为影响性能的配置参数（如 `BLOCK_SIZE_M`, `BLOCK_SIZE_N` 等）自动搜索最优值，这是实现硬件粒度性能优化的关键手段。
使用步骤如下：
1. **定义搜索空间 (Define Search Space)**:首先，定义一组希望 autotuner 探索的配置。
每个配置都是一个 `triton.Config` 对象。`triton.Config` 接受一个字典来定义你的自定义元参数（如块大小）。
```python
configs = [triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),    triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}), 
# ...可以根据需求添加更多配置
]
```
2. **应用装饰器 (Apply Decorator)**:将 `@triton.autotune` 装饰器应用到 `@triton.jit` kernel 之上。
你需要传入之前定义的 `configs` 列表，并指定 `key` 参数。`key` 是一个字符串列表，包含了函数签名中那些会影响性能的关键参数名称（通常是与问题规模相关的参数，如矩阵维度 M, N, K）。当这些 `key` 参数的输入值发生变化时，Triton 会重新运行基准测试来寻找新的最优配置。

3. 注意！！！调用 triton kernel 的时候不需要再传入configs中定义的变量，configs中的变量需要与kernel参数中最后若干个需要微调的参数保持一致

示例：

```python
configs = [
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}),
]
@triton.autotune(
    configs=configs,
    key=['M', 'N', 'K'],  # 当矩阵维度 M, N, K 变化时触发 auto-tune
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pass
def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty(M, N, device='cuda', dtype=torch.float32)
    
    # 网格：基于 BLOCK_M 和 BLOCK_N
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    # 调用内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```
