---
name: error-handling
description: "GPU代码错误处理和边界检查最佳实践"
category: implementation
version: "1.0.0"
license: MIT
---

# GPU代码错误处理

## 概述

良好的错误处理是生产级GPU代码的关键，包括编译时检查、运行时检查和调试支持。

## CUDA错误检查

### 基础宏定义

```cpp
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// 使用示例
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### Kernel启动错误检查

```cpp
// Kernel启动
my_kernel<<<grid, block>>>(args);

// 检查启动错误
CUDA_CHECK(cudaGetLastError());

// 检查执行错误
CUDA_CHECK(cudaDeviceSynchronize());
```

### C++异常封装

```cpp
class CUDAException : public std::runtime_exception {
public:
    CUDAException(cudaError_t error, const char* file, int line)
        : std::runtime_error(
            std::string("CUDA Error: ") + 
            cudaGetErrorString(error) +
            " at " + file + ":" + std::to_string(line)
        ) {}
};

#define CUDA_THROW(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw CUDAException(error, __FILE__, __LINE__); \
    } \
} while(0)

// 使用示例
try {
    CUDA_THROW(cudaMalloc(&d_data, size));
    my_kernel<<<grid, block>>>(d_data);
    CUDA_THROW(cudaDeviceSynchronize());
} catch (const CUDAException& e) {
    std::cerr << e.what() << std::endl;
    // 清理资源...
}
```

## Kernel内边界检查

### 基本边界检查

```cpp
__global__ void safe_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ✅ 边界检查
    if (idx < N) {
        data[idx] = process(data[idx]);
    }
}
```

### 2D边界检查

```cpp
__global__ void safe_2d_kernel(float* data, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // ✅ 2D边界检查
    if (row < M && col < N) {
        int idx = row * N + col;
        data[idx] = process(data[idx]);
    }
}
```

### 断言检查

```cpp
__global__ void kernel_with_asserts(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // 运行时断言（debug模式）
        assert(data[idx] >= 0.0f);
        assert(data[idx] <= 1.0f);
        
        data[idx] = process(data[idx]);
    }
}
```

## 数值稳定性检查

### NaN/Inf检查

```cpp
__device__ inline bool is_valid(float x) {
    return !isnan(x) && !isinf(x);
}

__global__ void safe_compute(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        float result = compute(input[idx]);
        
        // 检查数值有效性
        if (is_valid(result)) {
            output[idx] = result;
        } else {
            // 使用默认值或标记错误
            output[idx] = 0.0f;
            printf("Warning: invalid value at index %d\n", idx);
        }
    }
}
```

### 除零保护

```cpp
__device__ inline float safe_divide(float a, float b) {
    const float epsilon = 1e-8f;
    return a / (b + epsilon);
}

__device__ inline float safe_log(float x) {
    const float epsilon = 1e-8f;
    return logf(max(x, epsilon));
}
```

## 内存访问保护

### 检查内存访问模式

```cpp
__global__ void check_access_pattern(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // 对齐检查（仅在调试模式）
        #ifdef DEBUG
        if (reinterpret_cast<uintptr_t>(&data[idx]) % 16 != 0) {
            printf("Warning: unaligned access at %d\n", idx);
        }
        #endif
        
        data[idx] = process(data[idx]);
    }
}
```

### Shared Memory边界检查

```cpp
__global__ void safe_shared_memory(float* input, float* output, int N) {
    __shared__ float shared[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 检查shared memory索引
    if (tid < 256 && gid < N) {
        shared[tid] = input[gid];
    }
    __syncthreads();
    
    // 使用前再次检查
    if (tid < 256 && gid < N) {
        output[gid] = shared[tid];
    }
}
```

## Python/Triton错误处理

### 基本检查

```python
def safe_kernel_launch(kernel, grid, *args):
    """安全启动kernel"""
    try:
        kernel[grid](*args)
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"Kernel launch failed: {e}")
        raise
```

### 输入验证

```python
def validate_inputs(x, y):
    """验证输入tensor"""
    # 类型检查
    assert x.dtype == y.dtype, f"Type mismatch: {x.dtype} vs {y.dtype}"
    
    # 设备检查
    assert x.device == y.device, "Inputs must be on same device"
    
    # 形状检查
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
    
    # 连续性检查
    assert x.is_contiguous(), "Input x must be contiguous"
    assert y.is_contiguous(), "Input y must be contiguous"
    
    # NaN/Inf检查
    assert not torch.isnan(x).any(), "Input x contains NaN"
    assert not torch.isinf(x).any(), "Input x contains Inf"

def safe_vector_add(x, y):
    """带验证的向量加法"""
    validate_inputs(x, y)
    
    output = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output
```

## 调试辅助

### 条件编译

```cpp
#ifdef DEBUG
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)
#endif

__global__ void debug_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        DEBUG_PRINT("Thread %d processing data[%d] = %f\n", 
                    idx, idx, data[idx]);
        data[idx] = process(data[idx]);
    }
}
```

### 性能计数器

```cpp
__global__ void instrumented_kernel(float* data, int N, int* counters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // 计数特殊情况
        if (data[idx] < 0) {
            atomicAdd(&counters[0], 1);  // 负数计数
        }
        if (data[idx] == 0) {
            atomicAdd(&counters[1], 1);  // 零计数
        }
        
        data[idx] = process(data[idx]);
    }
}
```

## 错误恢复策略

### 重试机制

```python
def kernel_with_retry(kernel, grid, *args, max_retries=3):
    """带重试的kernel启动"""
    for attempt in range(max_retries):
        try:
            kernel[grid](*args)
            torch.cuda.synchronize()
            return  # 成功
        except RuntimeError as e:
            if attempt == max_retries - 1:
                raise  # 最后一次尝试失败
            print(f"Attempt {attempt + 1} failed, retrying...")
            torch.cuda.empty_cache()  # 清理缓存
```

### Fallback实现

```python
def robust_matmul(a, b):
    """带fallback的矩阵乘法"""
    try:
        # 尝试使用自定义kernel
        return custom_matmul_kernel(a, b)
    except RuntimeError as e:
        print(f"Custom kernel failed: {e}")
        print("Falling back to PyTorch implementation")
        return torch.matmul(a, b)
```

## 测试策略

### 边界情况测试

```python
def test_kernel_edge_cases(kernel_func):
    """测试边界情况"""
    # 空输入
    x = torch.tensor([], device='cuda')
    try:
        result = kernel_func(x)
        assert result.shape == x.shape
    except Exception as e:
        print(f"Empty input test failed: {e}")
    
    # 单元素
    x = torch.tensor([1.0], device='cuda')
    result = kernel_func(x)
    assert result.shape == (1,)
    
    # 大输入
    x = torch.randn(10**7, device='cuda')
    result = kernel_func(x)
    assert result.shape == x.shape
    
    # NaN输入
    x = torch.tensor([float('nan')], device='cuda')
    result = kernel_func(x)
    # 检查如何处理NaN
```

## 最佳实践总结

1. **总是检查CUDA API返回值**
2. **Kernel内做边界检查**
3. **使用断言验证假设**
4. **添加数值稳定性保护**
5. **提供详细的错误信息**
6. **使用条件编译控制调试代码**
7. **编写边界情况测试**
8. **提供fallback实现**

## 错误处理模板

```cpp
// 完整的错误处理模板
template<typename T>
__global__ void robust_kernel(
    T* input, T* output, int N,
    int* error_flags  // 用于报告错误
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 1. 边界检查
    if (idx >= N) return;
    
    // 2. 数值验证
    T value = input[idx];
    if (isnan(value) || isinf(value)) {
        atomicOr(&error_flags[0], 1);  // 标记错误
        output[idx] = 0;  // 默认值
        return;
    }
    
    // 3. 实际计算
    T result = compute(value);
    
    // 4. 结果验证
    if (isnan(result) || isinf(result)) {
        atomicOr(&error_flags[1], 1);
        output[idx] = 0;
        return;
    }
    
    // 5. 安全存储
    output[idx] = result;
}
```

## 相关Skill

- **上游**: coder-agent (生成需要错误处理的代码)
- **相关**: cuda-basics, triton-syntax (具体实现)

