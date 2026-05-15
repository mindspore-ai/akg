---
name: cuda-c-basics
description: "CUDA C 核心概念、内核结构和标准编程模式"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
  operator_patterns: "all"
---

# CUDA C 编程基础

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `__global__` 修饰的 C/C++ 函数，在 GPU 上并行执行
- **特点**: 每个内核实例处理数据的一个子集，通过线程索引区分
- **调用**: 使用 `<<<grid_size, block_size>>>` 语法从主机代码启动

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置，如 `(num_blocks_x, num_blocks_y)`
- **块**: 每个线程块包含的线程数，如 `block_size = 256`
- **关系**: `grid_size = ceil(total_elements / block_size)`
- **限制**: 每块最多 1024 线程（大多数 GPU）

### 线程层次
- **Grid**: 所有线程块的集合
- **Block**: 一组可以协作的线程（共享内存、同步）
- **Warp**: 32 个线程为一组并行执行（SIMT 执行模型）
- **Thread**: 最基本的执行单元

### 内存层次
- **全局内存 (Global Memory)**: 所有线程可访问，延迟高，容量大
- **共享内存 (Shared Memory)**: 块内线程共享，延迟低，容量有限（通常 48-164 KB/SM）
- **寄存器 (Registers)**: 每个线程私有，最快访问
- **常量内存 (Constant Memory)**: 只读，缓存优化
- **纹理内存 (Texture Memory)**: 只读，空间局部性优化

## 2. 标准内核结构（五步模式）

所有 CUDA C 内核都遵循相同的五步结构模式：

```cuda
__global__ void standard_kernel(
    float* output, float* input, int n_elements
) {
    // 1. 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 2. 边界检查
    if (idx < n_elements) {
        // 3. 加载数据
        float data = input[idx];
        
        // 4. 执行计算
        float result = compute_function(data);
        
        // 5. 存储结果
        output[idx] = result;
    }
}
```

### 内核启动方式

```cuda
void launch_kernel(float* input, float* output, int n_elements) {
    const int block_size = 256;
    const int num_blocks = (n_elements + block_size - 1) / block_size;
    
    kernel<<<num_blocks, block_size>>>(output, input, n_elements);
}
```

## 3. 全局索引计算

### 一维数据处理
```cuda
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

### 二维数据处理
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### 三维数据处理
```cuda
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
int z = blockIdx.z * blockDim.z + threadIdx.z;
```

### 网格配置

```cuda
// 一维网格
int block_size = 256;
int num_blocks = (n_elements + block_size - 1) / block_size;
kernel<<<num_blocks, block_size>>>(...);

// 二维网格（矩阵操作）
dim3 block_size(16, 16);
dim3 grid_size((N + 15) / 16, (M + 15) / 16);
kernel<<<grid_size, block_size>>>(...);

// 三维网格（体积数据）
dim3 block_size(8, 8, 8);
dim3 grid_size((X + 7) / 8, (Y + 7) / 8, (Z + 7) / 8);
kernel<<<grid_size, block_size>>>(...);
```

## 4. 边界处理

### 基本边界检查
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n_elements) {
    // 安全访问 input[idx]
    output[idx] = input[idx];
}
```

### 二维边界检查
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) {
    // 安全访问矩阵元素
    output[row * N + col] = input[row * N + col];
}
```

## 5. 内存管理模式

### 主机-设备数据传输
```cuda
// 分配设备内存
float* d_input, *d_output;
cudaMalloc(&d_input, size * sizeof(float));
cudaMalloc(&d_output, size * sizeof(float));

// 拷贝数据到设备
cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

// 启动内核
kernel<<<grid, block>>>(d_output, d_input, size);

// 拷贝结果回主机
cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

// 释放内存
cudaFree(d_input);
cudaFree(d_output);
```

## 6. PyTorch 集成方式

### ⚠️ 重要：使用 Python JIT 编译方式

在生成算子时，必须在 Python 模块中内嵌 CUDA C 代码，使用 `torch.utils.cpp_extension.load_inline` 进行 JIT 编译：

```python
import torch
from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void my_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor my_kernel_call(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    my_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""

cpp_src = "torch::Tensor my_kernel_call(torch::Tensor input);"

kernel_module = load_inline(
    name="my_cuda",
    cpp_sources=cpp_src,
    cuda_sources=source,
    functions=["my_kernel_call"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)
```

## 7. 最佳实践总结

### 编程原则
- **单一职责**: 每个内核只做一件事
- **参数简单**: 避免复杂的数据结构传递
- **边界检查**: 所有数组访问前必须检查边界
- **内存局部性**: 尽量访问相邻内存位置

### 常见错误避免
1. **越界访问**: 忘记边界检查导致运行时错误或结果异常
2. **内存泄漏**: 忘记调用 `cudaFree` 释放设备内存
3. **同步错误**: 共享内存操作缺少 `__syncthreads()`
4. **类型不匹配**: 主机和设备端数据类型不一致
5. **设备内存不足**: 块大小过大或数据量超出 GPU 内存

### ⚠️ 注意事项
- 生成的内核代码**不要包含任何测试代码片段**
- **禁止**使用 `printf()`、`throw std::runtime_error()` 等打印/异常语句
- 内核内**不使用** `malloc` / `new` 进行动态分配
