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

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
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

## 5. Autotune 使用教程（仅限静态 shape）

Autotune 通过自动 benchmark 多组配置参数，找到当前硬件和数据规模下的最优配置并缓存，免去手动调参。

### 适用场景

- **推荐使用**：输入 shape 固定或变化范围有限（静态 shape），如固定 batch size 的 MatMul、固定序列长度的 Attention 等
- **禁止使用**：输入 shape 频繁变化（动态 shape）。autotune 根据 `key` 参数缓存最佳 config，动态 shape 下每组新 shape 都会触发一次完整 benchmark，反而严重拖慢性能

### 强制规则

1. **必须写 `restore_value`**：列出 kernel 的**所有输出指针参数名**。autotune benchmark 会对每个 config 反复执行 kernel，`restore_value` 在每次迭代前保存输出张量副本、迭代后恢复原值，防止不同 config 之间的结果互相污染。**不写 `restore_value` 会导致验证失败。**
2. **调用时不传 configs 参数**：autotune 自动传入。
3. **configs 参数必须是 constexpr**：在 kernel 中声明为 `PARAM: tl.constexpr`。
4. **key 参数**：指定哪些输入维度变化时重新 autotune。
5. **Ascend 不支持调优**：不要对 num_warps、num_ctas、num_stages 等参数进行修改调优，当前 Ascend 后端不支持。

### 标准写法

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 512}),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # ⚠ 必须：列出所有输出指针参数名
)
@triton.jit
def kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # kernel 实现
    pass

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        except:
            self.VEC_CORE_NUM = 40

    def forward(self, x):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = (self.VEC_CORE_NUM,)
        kernel[grid](x, output, n_elements)
        return output
```