# CUDA C 编程基础教程

本文档介绍 CUDA C 的核心概念和标准编程模式，通过详细示例帮助理解如何构建内核。

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `__global__` 修饰的 C/C++ 函数，在GPU上并行执行
- **特点**: 每个内核实例处理数据的一个子集，通过线程索引区分

### 网格 (Grid) 与块 (Block)
- **网格**: 内核启动时的并行维度配置，如 `(num_blocks_x, num_blocks_y)`
- **块**: 每个程序实例包含的线程数，如 `block_size = 256`
- **关系**: `grid_size = ceil(total_elements / block_size)`

### 内存层次
- **全局内存**: 所有线程可访问，延迟高，容量大
- **共享内存**: 块内线程共享，延迟低，容量有限
- **寄存器**: 每个线程私有，最快访问

## 2. 标准内核结构

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

## 3. 编程模式

### 3.1 向量操作模式
适用于元素级运算：加法、乘法、激活函数等。

```cuda
__global__ void vector_add_kernel(
    float* a, float* b, float* c, int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### 3.2 归约模式
适用于求和、最大值、最小值等聚合操作。

```cuda
__global__ void reduction_kernel(
    float* input, float* output, int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // 使用共享内存进行块内归约
        extern __shared__ float sdata[];
        sdata[threadIdx.x] = input[idx];
        __syncthreads();
        
        // 块内归约
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        // 第一个线程写入结果
        if (threadIdx.x == 0) {
            atomicAdd(output, sdata[0]);
        }
    }
}
```

### 3.3 矩阵乘法模式
使用2D线程配置处理矩阵数据。

```cuda
__global__ void matmul_kernel(
    float* A, float* B, float* C, 
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

## 4. 边界处理示例

### 基本边界检查
```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < n_elements) {
    // 安全访问 input[idx]
    output[idx] = input[idx];
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
