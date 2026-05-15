---
name: cuda-basics
description: "CUDA编程基础知识，包括内存模型、线程层次和常用优化技巧"
category: dsl
version: "1.0.0"
license: MIT
---

# CUDA编程基础

## 概述

CUDA (Compute Unified Device Architecture) 是NVIDIA推出的并行计算平台和编程模型。

## 线程层次结构

### 三层结构

```
Grid (网格)
  └─ Block (块) 
      └─ Thread (线程)
```

### 维度表示

```cpp
// 1D
dim3 block(256);
dim3 grid((N + 255) / 256);

// 2D
dim3 block(16, 16);
dim3 grid((M + 15) / 16, (N + 15) / 16);

// 3D
dim3 block(8, 8, 8);
dim3 grid((M + 7) / 8, (N + 7) / 8, (K + 7) / 8);
```

### 线程索引计算

```cpp
// 1D索引
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 2D索引
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

// 全局1D索引（从2D）
int idx = row * width + col;
```

## 内存层次

### 内存类型对比

| 内存类型 | 位置 | 访问速度 | 大小 | 作用域 | 生命周期 |
|---------|------|---------|------|-------|---------|
| Register | 片上 | 最快 | ~64KB/SM | Thread | Thread |
| Shared Memory | 片上 | 快 | 48KB-164KB | Block | Block |
| L1 Cache | 片上 | 快 | 128KB | - | - |
| L2 Cache | 片上 | 中 | MB级 | - | - |
| Global Memory | DRAM | 慢 | GB级 | Grid | Application |
| Constant Memory | DRAM | 中(有cache) | 64KB | Grid | Application |
| Texture Memory | DRAM | 中(有cache) | - | Grid | Application |

### 声明方式

```cpp
// Register (自动)
int local_var;

// Shared Memory
__shared__ float shared_data[256];

// Global Memory
__global__ void kernel(float* global_data) { }

// Constant Memory
__constant__ float const_data[1024];
```

## 内存访问优化

### 1. 合并内存访问 (Coalesced Access)

```cpp
// ✅ 好：连续访问
__global__ void coalesced_read(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // 线程连续访问
}

// ❌ 差：跨步访问
__global__ void strided_read(float* data, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    float val = data[idx];  // 线程跨步访问
}
```

### 2. 使用Shared Memory

```cpp
__global__ void use_shared_memory(float* input, float* output) {
    __shared__ float tile[TILE_SIZE];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 从全局内存加载到共享内存
    tile[tid] = input[gid];
    __syncthreads();  // 同步
    
    // 从共享内存读取（快速）
    float val = tile[tid];
    
    // 处理...
    output[gid] = val;
}
```

### 3. 避免Bank Conflict

Shared memory分为32个bank，同时访问同一bank的不同地址会导致冲突。

```cpp
// ❌ 有Bank Conflict
__shared__ float data[32][32];
float val = data[threadIdx.x][threadIdx.y];  // 列访问导致冲突

// ✅ 无Bank Conflict（通过padding）
__shared__ float data[32][33];  // 多一列padding
float val = data[threadIdx.x][threadIdx.y];
```

## 同步机制

### Block内同步

```cpp
__syncthreads();  // 等待block内所有线程到达此点
```

### Warp内同步（CUDA 9+）

```cpp
__syncwarp();  // 等待warp内所有线程
```

### 原子操作

```cpp
atomicAdd(&counter, 1);      // 原子加
atomicMax(&max_val, val);    // 原子最大值
atomicCAS(&lock, 0, 1);      // Compare-and-Swap
```

## 常用优化技巧

### 1. 循环展开

```cpp
// 手动展开
#pragma unroll
for (int i = 0; i < 4; ++i) {
    sum += data[i];
}

// 完全展开
sum = data[0] + data[1] + data[2] + data[3];
```

### 2. Warp Shuffle

在warp内线程间交换数据，无需共享内存：

```cpp
// Warp reduce
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

### 3. 向量化加载

```cpp
// 使用float4一次加载4个float
float4 val = reinterpret_cast<float4*>(data)[idx];
```

## 配置参数选择

### Block Size选择

```cpp
// 经验法则
// - Warp大小的倍数（32）
// - 考虑register和shared memory限制
// - 常用: 128, 256, 512

dim3 block(256);  // 常见选择
```

### Grid Size计算

```cpp
// 向上取整
int grid_size = (N + block_size - 1) / block_size;
```

### Occupancy优化

```cpp
// 使用occupancy calculator
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize,
    my_kernel, 0, 0
);
```

## 调试技巧

### 1. 检查错误

```cpp
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(1); \
    } \
}

CUDA_CHECK(cudaMalloc(&d_data, size));
```

### 2. printf调试

```cpp
__global__ void debug_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Debug: value = %d\n", value);
    }
}
```

### 3. CUDA-GDB

```bash
cuda-gdb ./program
(cuda-gdb) break my_kernel
(cuda-gdb) run
(cuda-gdb) cuda thread
```

## 完整示例：向量加法

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    
    // 主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }
    
    // 设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    // 拷贝到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 启动kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vector_add<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    
    // 拷贝回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}
```

## 性能分析

### 理论性能

```python
# 计算理论FLOPS
peak_flops = num_sms * clock_mhz * ops_per_clock * 1e6

# 计算理论带宽
peak_bandwidth_gb_s = memory_bus_width / 8 * memory_clock_mhz * 2 / 1000
```

### Roofline Model

```
Performance = min(
    Peak_FLOPS,
    Peak_Bandwidth * Arithmetic_Intensity
)
```

## 常见陷阱

1. **忘记同步**: 使用shared memory前后需要`__syncthreads()`
2. **Race Condition**: 多个线程写同一地址需要原子操作
3. **Bank Conflict**: 注意shared memory访问模式
4. **Divergence**: 避免warp内的分支divergence
5. **Occupancy过低**: 检查register和shared memory使用

## 参考资料

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Professional CUDA C Programming](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)

