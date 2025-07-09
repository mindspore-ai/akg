# Triton 编程实例指南 (精简版)

本文档通过实例展示 Triton 的核心编程模式。

## 1. Triton 内核基本结构

所有 Triton 内核都遵循相似的结构：`获取ID -> 计算偏移 -> 创建掩码 -> 加载 -> 计算 -> 存储`。
以下是一个典型的 1D 内核，用于向量操作。

```python
@triton.jit
def vector_op_kernel(
    output_ptr, input_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    # 获取当前程序块要处理的数据范围
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 带掩码地加载、计算、存储
    data = tl.load(input_ptr + offsets, mask=mask)
    result = data * 2
    tl.store(output_ptr + offsets, result, mask=mask)
```

### 启动函数
```python
def launch_kernel(input_tensor, output_tensor):
    # 选择块大小 (通常是2的幂)
    BLOCK_SIZE = 1024
    # 计算网格大小
    grid = (triton.cdiv(input_tensor.numel(), BLOCK_SIZE),)
    # 启动内核
    vector_op_kernel[grid](
        output_tensor, input_tensor, input_tensor.numel(), 
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

## 2. 核心编程模式

基于上述结构，可以组合出更复杂的操作。

### 归约 (Reduction)
归约操作通常在块内完成，然后可以选择在块间进行二次归约。

```python
@triton.jit
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # --- 此处省略 pid, offsets, mask 的计算 ---
    # 假设已加载数据 `data`
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 块内归约
    block_sum = tl.sum(data, axis=0)
    
    # 原子操作用于将块结果安全地写回全局内存
    tl.atomic_add(output_ptr, block_sum)
```

### 矩阵乘法 (Matrix Multiplication)
矩阵乘法是Triton的核心应用。推荐使用**块指针 (`tl.make_block_ptr`)** 来处理2D数据，代码更简洁高效。

以下是 Matmul 内核中**核心循环**的简化逻辑：

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
    # --- 省略 pid_m, pid_n 的计算 ---

    # 1. 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 2. 主循环 (沿K轴迭代)
    for k in range(0, K, BLOCK_SIZE_K):
        # 3. 使用 `make_block_ptr` 创建指针 (推荐做法)
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
        
        # 4. 加载数据块
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        # 5. 核心计算：矩阵乘加
        accumulator += tl.dot(a, b)
    
    # --- 省略最终结果的存储 ---
```

## 3. 性能与技巧

- **内存合并**: 性能关键。确保`tl.load`/`tl.store`访问连续内存。
- **块大小**: 选择2的幂（如128, 256, 1024），平衡并行度与资源。
- **边界处理**: 始终用 `mask` 或块指针的 `boundary_check` 来防止越界。
- **使用块指针**: 对于2D数据，`tl.make_block_ptr` 是比手动计算指针更优的选择。
- **数值稳定性**: 对于 `softmax` 等，通过 `x_stable = x - tl.max(x)` 来防止数值溢出。 