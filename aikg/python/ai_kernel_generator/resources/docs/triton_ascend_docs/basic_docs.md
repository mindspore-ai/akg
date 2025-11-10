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
def launch_kernel(input_tensor):
    output_tensor = torch.empty(size, dtype, device)
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
适用于矩阵乘法等多维块计算,使用固定核心数启动。

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 关键:使用固定核心数启动,每个核心处理多个块
    pid = tl.program_id(0)  # 核心ID: 0~num_cores-1
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N

    # 每个核心循环处理多个块
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        # 计算当前块的2D索引
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        # 初始化累加器
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K维度循环
        for k in range(0, K, BLOCK_K):
            # 加载A块
            a_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * K + \
                       (k + tl.arange(0, BLOCK_K))[None, :]
            a_mask = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M
            a = tl.load(a_ptr + a_offset, mask=a_mask, other=0.0)

            # 加载B块
            b_offset = (k + tl.arange(0, BLOCK_K))[:, None] * N + \
                       (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
            b_mask = (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N
            b = tl.load(b_ptr + b_offset, mask=b_mask, other=0.0)

            # 矩阵乘累加
            accumulator += tl.dot(a, b)

        # 存储结果
        c_offset = (block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * N + \
                   (block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
        c_mask = ((block_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < M) & \
                 ((block_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :] < N)
        tl.store(c_ptr + c_offset, accumulator, mask=c_mask)

def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    num_cores = 20  # Ascend 910B4有20个AI Core
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

    # 关键:固定核心数启动,grid=(num_cores,)不是(NUM_BLOCKS,)
    matmul_kernel[(num_cores,)](
        a, b, c, M, N, K, num_cores,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return c
```

**关键点**:
- ✅ 使用 `grid=(num_cores,)` 固定启动核心数(如20个)
- ✅ 每个核心通过 `for block_idx in range(pid, NUM_BLOCKS, num_cores)` 循环处理多个块
- ❌ 不要使用 `grid=(NUM_BLOCKS_M * NUM_BLOCKS_N,)` 为每个块启动一个程序

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

## 5. autotune使用教程

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

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),) # grid设置是tuple
    
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

### 关键要点

1. **grid必须使用lambda**：`grid = lambda meta: (...)`
3. **不要传递configs参数**：调用kernel时不要传`BLOCK_SIZE`等autotune的参数
4. **configs参数必须是constexpr**：在kernel中声明为`PARAM: tl.constexpr`
5. **key参数**：指定哪些输入维度变化时重新autotune

**注意** 不要对'num_warps', 'num_ctas', 'num_stages', 'num_buffers_warp_spec', 'num_consumer_groups', 'reg_dec_producer', ，'reg_inc_consumer', 'maxnreg'进行修改调优，当前Ascend后端不支持