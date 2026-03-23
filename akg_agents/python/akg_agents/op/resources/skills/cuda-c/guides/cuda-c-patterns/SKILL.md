---
name: cuda-c-patterns
description: "CUDA C 三大编程模式：向量操作、归约、矩阵乘法"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
  operator_patterns: "elementwise, reduce, matmul"
---

# CUDA C 编程模式

## 1. 向量操作模式

适用于元素级运算：加法、乘法、激活函数等。

### 标准代码结构

```cuda
__global__ void vector_add_kernel(
    const float* a, const float* b, float* c, int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### 适用算子
- 算术运算: add, mul, sub, div
- 激活函数: relu, sigmoid, tanh, gelu, silu
- 数学函数: exp, log, sqrt, pow, abs
- 类型转换: cast
- 广播操作: broadcast

### 关键要点
- 使用一维索引 `blockIdx.x * blockDim.x + threadIdx.x`
- 边界检查 `if (idx < n_elements)`
- 简单直接的数据流：加载 → 计算 → 存储
- 推荐块大小: 256 或 512

### ReLU 示例

```cuda
__global__ void relu_kernel(
    const float* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(input[idx], 0.0f);
    }
}
```

### GELU 示例

```cuda
__global__ void gelu_kernel(
    const float* input, float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        // 近似 GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}
```

### 多输入逐元素操作

```cuda
__global__ void fused_multiply_add_kernel(
    const float* a, const float* b, const float* c,
    float* output, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = a[idx] * b[idx] + c[idx];
    }
}
```

## 2. 归约模式

适用于求和、最大值、最小值等聚合操作。

### 标准代码结构（共享内存归约）

```cuda
__global__ void reduction_sum_kernel(
    const float* input, float* output, int n_elements
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 加载数据到共享内存
    sdata[tid] = (idx < n_elements) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 块内归约（树形归约）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 第一个线程写入块级结果
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}
```

### 适用算子
- 基础归约: sum, mean, max, min, prod
- 归一化: softmax, logsoftmax, layernorm, batchnorm, rmsnorm
- 统计: variance, std
- 搜索: argmax, argmin

### 关键要点
- 使用 `extern __shared__` 声明动态共享内存
- 树形归约（每次折半）
- `__syncthreads()` 确保数据一致性
- `atomicAdd` 用于跨 block 的全局归约
- ⚠️ 同步点必须所有线程都能到达

### Softmax 示例（数值稳定版）

```cuda
__global__ void softmax_kernel(
    const float* input, float* output, int rows, int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // 1. 找最大值（数值稳定）
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_input[i]);
    }
    
    // Warp 内归约最大值
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xFFFFFFFF, max_val, offset));
    }
    
    // 通过共享内存跨 warp 归约
    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = max_val;
    __syncthreads();
    max_val = s_max;
    
    // 2. 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __expf(row_input[i] - max_val);
    }
    
    // 归约 sum
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    sum = s_sum;
    
    // 3. 计算 softmax
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] = __expf(row_input[i] - max_val) / sum;
    }
}
```

### LayerNorm 示例

```cuda
__global__ void layer_norm_kernel(
    const float* input, float* output,
    const float* gamma, const float* beta,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    
    const float* row_input = input + row * cols;
    float* row_output = output + row * cols;
    
    // 1. 计算均值
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += row_input[i];
    }
    // warp 归约
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    __shared__ float s_mean;
    if (threadIdx.x == 0) s_mean = sum / cols;
    __syncthreads();
    float mean = s_mean;
    
    // 2. 计算方差
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float diff = row_input[i] - mean;
        var_sum += diff * diff;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        var_sum += __shfl_down_sync(0xFFFFFFFF, var_sum, offset);
    }
    __shared__ float s_var;
    if (threadIdx.x == 0) s_var = var_sum / cols;
    __syncthreads();
    float rstd = rsqrtf(s_var + eps);
    
    // 3. 归一化
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float normalized = (row_input[i] - mean) * rstd;
        row_output[i] = normalized * gamma[i] + beta[i];
    }
}
```

## 3. 矩阵乘法模式

适用于矩阵乘法等多维块计算。

### 标准代码结构（朴素版）

```cuda
__global__ void matmul_kernel(
    const float* A, const float* B, float* C,
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

### 共享内存优化版

```cuda
#define TILE_SIZE 16

__global__ void matmul_shared_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 加载到共享内存
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        
        // 计算部分乘积
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 适用算子
- 矩阵运算: matmul, bmm (batch matmul), linear
- 卷积: conv2d, conv3d（im2col 变换后）
- 注意力机制: attention

### 关键要点
- **2D Grid**: 使用 `dim3` 配置二维网格和线程块
- **共享内存优化**: 分块加载减少全局内存访问
- **同步**: 每次分块加载后 `__syncthreads()`
- **边界处理**: 越界位置填充 0
- **分块大小**: 常用 16x16 或 32x32

### Host 侧启动

```cuda
// 朴素版启动
dim3 block(16, 16);
dim3 grid((N + 15) / 16, (M + 15) / 16);
matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);

// 共享内存版启动
dim3 block(TILE_SIZE, TILE_SIZE);
dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
matmul_shared_kernel<<<grid, block>>>(A, B, C, M, N, K);
```

## 模式选择指南

| 算子类型 | 推荐模式 | 关键特征 | 块大小 |
|---------|---------|---------|-------|
| Element-wise | 向量操作 | 逐元素独立计算 | 256/512 |
| Reduction | 归约模式 | 需要聚合多个值 | 256 |
| MatMul/Conv | 矩阵乘法 | 多维块计算，2D Grid | 16x16/32x32 |
| Softmax/Norm | 归约+元素操作 | 行级归约+逐元素 | 256 |
| Attention | 组合模式 | MatMul + Softmax | 视情况而定 |

## 最佳实践

1. **选择合适的模式**: 根据算子特性选择基础模式
2. **优化块大小**: 平衡并行度和资源占用
3. **注意边界**: 使用 `if (idx < n)` 处理不规则形状
4. **数值稳定性**: 归约类算子先减最大值防溢出
5. **内存访问**: 保证合并访问，使用共享内存缓存
6. **同步安全**: `__syncthreads()` 必须所有线程到达
7. **减少原子操作**: 先块内归约再全局写回
