---
name: tilelang-cuda-examples-torch
description: "PyTorch + TileLang CUDA 完整示例代码"
category: example
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
  framework: torch
  examples: "matmul, elementwise, layernorm, gemv, flash_attention"
---

# PyTorch + TileLang CUDA 示例代码

本 Skill 包含完整的可运行示例代码，展示如何在 PyTorch 中使用 TileLang CUDA 编写高性能 kernel。

## 示例列表

### 1. 矩阵乘法（GEMM）
**算子类型**: MatMul
**关键点**:
- 共享内存缓存输入块
- `T.gemm` 利用 Tensor Core
- 软件流水线 `T.Pipelined`
- 混合精度（float32 累加器）

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def matmul(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(A: T.Tensor((M, K), "float16"),
             B: T.Tensor((K, N), "float16"),
             C: T.Tensor((M, N), "float16")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            T.clear(C_local)
            
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * block_M, bx * block_N])
    
    return main

# 调用方式
def matmul_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    block_M, block_N, block_K = 128, 128, 32
    kernel = matmul(M, N, K, block_M, block_N, block_K)
    C = kernel(A, B)  # out_idx=[-1]，只传输入
    return C
```

### 2. 矩阵乘法（float32，手动管理输出）
**算子类型**: MatMul
**关键点**:
- 不使用 `out_idx`，手动管理输出
- float32 数据类型
- 需要手动创建输出张量并一起传入

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit
def square_matrix_multiply(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
            A: T.Tensor((M, K), "float32"),
            B: T.Tensor((K, N), "float32"),
            C: T.Tensor((M, N), "float32")):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float32")
            B_shared = T.alloc_shared((block_K, block_N), "float32")
            C_local = T.alloc_fragment((block_M, block_N), "float")
            
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

def square_matrix_multiply_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N = A.size(0)
    block_M, block_N, block_K = 128, 128, 32
    # 不使用 out_idx 时，需要手动创建输出张量
    C = torch.empty_like(A)
    kernel = square_matrix_multiply(N, N, N, block_M, block_N, block_K)
    kernel(A, B, C)  # 传入所有参数包括输出
    return C
```

### 3. 逐元素操作（Element-wise Add）
**算子类型**: Element-wise
**关键点**:
- 最简单的 TileLang 内核示例
- 使用 `T.Parallel` 进行并行计算
- 直接访问全局内存

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def elementwise_add(M, N, block_M, block_N, threads):
    @T.prim_func
    def main(A: T.Tensor((M, N), "float32"),
             B: T.Tensor((M, N), "float32"),
             C: T.Tensor((M, N), "float32")):
        
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = by * block_M + local_y
                x = bx * block_N + local_x
                C[y, x] = A[y, x] + B[y, x]
    
    return main

# 调用方式
def add_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, N = A.shape
    block_M, block_N = 32, 32
    threads = 256
    kernel = elementwise_add(M, N, block_M, block_N, threads)
    return kernel(A, B)
```

### 4. LayerNorm（层归一化）
**算子类型**: Reduce + Element-wise
**关键点**:
- 使用 `out_idx=[-1]` 指定输出
- `T.reduce_sum` 内置归约（避免手动同步）
- 边界检查处理非对齐数据
- float32 中间计算保证精度

```python
import tilelang as tl
import tilelang.language as T
import torch

@tl.jit(out_idx=[-1])
def layer_norm_kernel(batch_size, features, dim1, dim2, block_size):
    @T.prim_func
    def main(x: T.Tensor((batch_size, features, dim1, dim2), "float16"),
             y: T.Tensor((batch_size, features, dim1, dim2), "float16")):
        
        total_size = features * dim1 * dim2

        with T.Kernel(batch_size, T.ceildiv(total_size, block_size), threads=block_size) as (sample_idx, bx):
            A_shared = T.alloc_shared((block_size,), "float32")
            A_pow_local = T.alloc_fragment((block_size,), "float32")
            A_powsum = T.alloc_fragment((1,), "float32")
            
            # 数据加载和计算
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid
                
                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")
                    
                    A_shared[tid] = input_val
                    A_pow_local[tid] = input_val * input_val
                else:
                    A_shared[tid] = 0.0
                    A_pow_local[tid] = 0.0
            
            # ✅ 使用内置归约，避免同步和线程卡死
            T.reduce_sum(A_pow_local, A_powsum, dim=0)
            
            # 计算归一化因子并应用
            for tid in T.Parallel(block_size):
                elem_idx = bx * block_size + tid
                
                if elem_idx < total_size:
                    c = elem_idx // (dim1 * dim2)
                    h = (elem_idx % (dim1 * dim2)) // dim2
                    w = elem_idx % dim2
                    input_val = x[sample_idx, c, h, w].astype("float32")
                    
                    mean_val = A_powsum[0] / total_size
                    var_val = A_powsum[0] / total_size - mean_val * mean_val
                    normalized = (input_val - mean_val) / T.sqrt(var_val + 1e-5)
                    y[sample_idx, c, h, w] = normalized.astype("float16")

    return main

# 调用方式
def layer_norm(input_tensor: torch.Tensor):
    batch_size, features, dim1, dim2 = input_tensor.shape
    block_size = 256
    kernel = layer_norm_kernel(batch_size, features, dim1, dim2, block_size)
    y = kernel(input_tensor)  # out_idx=[-1]，自动创建输出
    return y
```

### 5. GEMV（矩阵向量乘法）
**算子类型**: GEMV
**关键点**:
- 使用 `T.Parallel` 获取线程索引
- 使用 `T.serial` 进行串行循环
- 使用 `T.alloc_local` 进行线程私有累加
- 类型转换 `.astype("float")` 保证精度

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def gemv(N, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def main(A: T.Tensor((K,), "float16"),
             B: T.Tensor((N, K), "float16"),
             C: T.Tensor((N,), "float16")):
        
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            A_shared = T.alloc_shared((BLOCK_K,), "float16")
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), "float16")
            
            for tn in T.Parallel(BLOCK_N):
                C_reg = T.alloc_local((1,), "float")
                T.clear(C_reg)
                
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for tk in T.serial(BLOCK_K):
                        A_shared[tk] = A[bk * BLOCK_K + tk]
                        B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                    
                    for tk in T.serial(BLOCK_K):
                        C_reg[0] += A_shared[tk].astype("float") * B_shared[tn, tk].astype("float")
                
                C[bn * BLOCK_N + tn] = C_reg[0]
    
    return main

# 调用方式
def gemv_call(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    N, K = B.shape
    BLOCK_N, BLOCK_K = 128, 32
    kernel = gemv(N, K, BLOCK_N, BLOCK_K)
    return kernel(A, B)
```

## 通用模式

所有 TileLang CUDA 示例都遵循相同的结构：

### 内核定义
```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def kernel_name(shape_params, block_params):
    @T.prim_func
    def main(input1: T.Tensor(shape, dtype),
             input2: T.Tensor(shape, dtype),
             output: T.Tensor(shape, dtype)):
        
        with T.Kernel(grid_x, grid_y, threads=N) as (bx, by):
            # 1. 内存分配
            shared = T.alloc_shared(shape, dtype)
            local = T.alloc_fragment(shape, dtype)
            
            # 2. 数据加载和计算
            T.copy(input[...], shared)
            # ... 计算逻辑 ...
            
            # 3. 结果写回
            T.copy(local, output[...])
    
    return main
```

### 调用函数
```python
def call_function(input_tensor: torch.Tensor) -> torch.Tensor:
    # 确定形状和分块参数
    M, N = input_tensor.shape
    block_M, block_N = 128, 128
    
    # 编译内核
    kernel = kernel_name(M, N, block_M, block_N)
    
    # 使用 out_idx 时：只传输入
    result = kernel(input_tensor)
    
    # 不使用 out_idx 时：手动创建输出
    # output = torch.empty_like(input_tensor)
    # kernel(input_tensor, output)
    
    return result
```

## 关键注意事项

### 1. out_idx 使用规范
```python
# ✅ 使用 out_idx：只传输入，输出自动创建
@tilelang.jit(out_idx=[-1])
result = kernel(input_data)

# ✅ 不使用 out_idx：手动管理所有张量
@tilelang.jit
output = torch.empty_like(input_data)
kernel(input_data, output)

# ❌ 错误：使用 out_idx 但额外传输出
@tilelang.jit(out_idx=[-1])
output = torch.empty_like(input_data)
kernel(input_data, output)  # ValueError!
```

### 2. 张量设备和数据类型
```python
# 确保输入在 CUDA 设备上
input_tensor = input_tensor.cuda()

# 类型转换在内核内完成
input_val = x[i].astype("float32")  # float16 -> float32
result = normalized.astype("float16")  # float32 -> float16
```

### 3. 内置归约替代手动同步
```python
# ✅ 正确：使用内置归约
T.reduce_sum(input_local, output_local, dim=0)

# ❌ 错误：手动归约（会导致死锁）
# while stride > 0:
#     if tid < stride:
#         shared[tid] += shared[tid + stride]
#     T.sync_threads()
#     stride //= 2
```

## 验证正确性

```python
# 与 PyTorch 原生实现对比
x = torch.randn(128, 256, device='cuda', dtype=torch.float16)
output_tilelang = kernel_call(x)
output_torch = torch_reference(x)

# 检查差异
diff = (output_tilelang - output_torch).abs().max()
print(f"Max difference: {diff.item()}")
assert diff < 1e-3, "Results mismatch!"
```
