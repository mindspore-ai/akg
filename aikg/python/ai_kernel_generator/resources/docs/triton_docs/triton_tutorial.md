# Triton 编程指南

Triton 是一种用于编写高效 GPU 内核的领域特定语言，专为深度学习应用优化。

## 基本概念

### 程序和块
- Triton 内核以"程序"（program）为单位执行
- 每个程序处理一个数据块（block）
- 程序可以在多个维度上并行执行

### 内存层次
- **全局内存**：GPU 的主内存，通过指针访问
- **共享内存**：Triton 自动管理，用户无需显式处理
- **寄存器**：存储临时计算结果

## 基本结构

### 典型的 Triton 内核
```python
@triton.jit
def kernel_name(
    output_ptr,           # 输出指针
    input_ptr,            # 输入指针  
    n_elements,           # 元素数量
    BLOCK_SIZE: tl.constexpr,  # 编译时常量
):
    # 1. 获取程序ID和偏移
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 2. 创建指针和掩码
    mask = offsets < n_elements
    input_ptrs = input_ptr + offsets
    
    # 3. 加载数据
    data = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # 4. 执行计算
    result = compute_function(data)
    
    # 5. 存储结果
    output_ptrs = output_ptr + offsets
    tl.store(output_ptrs, result, mask=mask)
```

### 启动函数
```python
def launch_kernel(input_tensor, output_tensor):
    n_elements = input_tensor.numel()
    
    # 选择块大小（通常是2的幂）
    BLOCK_SIZE = 1024
    
    # 计算网格大小
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动内核
    kernel_name[grid](
        output_tensor,
        input_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
```

## 常见模式

### 1. 向量操作
```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. 归约操作
```python
@triton.jit  
def reduction_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(data)  # 块内归约
    
    # 存储归约结果
    if pid == 0:
        tl.store(output_ptr, result)
```

### 3. 矩阵操作
```python
@triton.jit
def matrix_kernel(a_ptr, b_ptr, c_ptr, M, N, K, 
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算块偏移
    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # 创建指针
    a_ptrs = a_ptr + (rm[:, None] * stride_am + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_ak)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE_K)[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # 执行矩阵乘法
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # 加载数据块
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # 累加
        accumulator += tl.dot(a, b)
        # 更新指针
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # 存储结果
    c_ptrs = c_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)
```

## 性能优化技巧

### 1. 块大小选择
- 通常选择2的幂（128, 256, 512, 1024）
- 考虑内存使用和并行度的平衡
- 使用 `triton.next_power_of_2()` 自动选择

### 2. 内存访问优化
- 确保内存访问是合并的（连续的）
- 使用适当的步幅（stride）
- 避免不必要的内存访问

### 3. 掩码使用
- 正确使用掩码处理边界条件
- 为 `other` 参数选择合适的默认值
- 避免过度使用掩码影响性能

### 4. 数值稳定性
```python
# 对于 softmax 等操作，减去最大值防止溢出
x_max = tl.max(x, axis=1)
x_stable = x - x_max[:, None]
```

## 常见错误和解决方案

### 1. 越界访问
- 问题：访问超出数组边界的内存
- 解决：使用正确的掩码和边界检查

### 2. 类型不匹配
- 问题：数据类型不一致导致计算错误
- 解决：确保所有操作使用一致的数据类型

### 3. 块大小不当
- 问题：块大小不是2的幂或过大/过小
- 解决：选择合适的块大小并进行性能测试 