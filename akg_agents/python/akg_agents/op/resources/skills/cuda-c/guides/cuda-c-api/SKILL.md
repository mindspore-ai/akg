---
name: cuda-c-api
description: "CUDA C 编程接口完整参考手册"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
---

# CUDA C API 参考手册

本文档提供 CUDA C 核心编程接口的详细参考，包括函数签名、参数说明和使用示例。

## 1. 函数修饰符

### __global__
```cuda
__global__ void kernel_function(参数列表);
```
- **功能**: 标记为可在 GPU 上执行的内核函数
- **调用**: 只能从主机代码调用，使用 `<<<>>>` 语法启动
- **返回值**: 必须为 `void`

### __device__
```cuda
__device__ float device_function(参数列表);
```
- **功能**: 标记为在 GPU 上执行的设备函数
- **调用**: 只能从其他 `__device__` 或 `__global__` 函数调用
- **用途**: 内核内部的辅助函数

### __host__
```cuda
__host__ void host_function(参数列表);
```
- **功能**: 标记为在 CPU 上执行的函数（默认）
- **调用**: 只能从主机代码调用

### __host__ __device__
```cuda
__host__ __device__ float utility_function(float x);
```
- **功能**: 同时在 CPU 和 GPU 上可用
- **用途**: 通用工具函数

## 2. 内存类型修饰符

### __shared__
```cuda
__shared__ float shared_memory[256];
```
- **功能**: 声明线程块内共享的内存
- **生命周期**: 与线程块相同
- **访问**: 块内所有线程可读写
- **容量**: 通常 48-164 KB/SM

### __constant__
```cuda
__constant__ float constant_data[64];
```
- **功能**: 声明只读的常量内存
- **特点**: 缓存优化，适合广播读取
- **设置**: 通过 `cudaMemcpyToSymbol` 从主机端设置

### extern __shared__
```cuda
extern __shared__ float dynamic_shared[];
```
- **功能**: 动态分配的共享内存
- **大小**: 在内核启动时通过第三个参数指定
- **启动**: `kernel<<<grid, block, shared_mem_bytes>>>(args)`

## 3. 内核启动语法

### 基本语法
```cuda
kernel_name<<<grid_size, block_size>>>(参数列表);
kernel_name<<<grid_size, block_size, shared_mem_bytes>>>(参数列表);
kernel_name<<<grid_size, block_size, shared_mem_bytes, stream>>>(参数列表);
```
- **grid_size**: 网格大小（`int` 或 `dim3`）
- **block_size**: 线程块大小（`int` 或 `dim3`）
- **shared_mem_bytes**: 动态共享内存大小（可选，默认 0）
- **stream**: CUDA 流（可选，默认 0）

### dim3 类型
```cuda
dim3 grid_size(blocks_x, blocks_y, blocks_z);
dim3 block_size(threads_x, threads_y, threads_z);
```
- **用途**: 多维网格和线程块配置
- **默认**: 未指定的维度默认为 1

## 4. 线程和块索引系统

### 块索引变量
```cuda
int bx = blockIdx.x;   // X 方向块索引
int by = blockIdx.y;   // Y 方向块索引
int bz = blockIdx.z;   // Z 方向块索引
```
- **类型**: `uint3`
- **用途**: 确定当前线程块在网格中的位置

### 线程索引变量
```cuda
int tx = threadIdx.x;  // X 方向线程索引
int ty = threadIdx.y;  // Y 方向线程索引
int tz = threadIdx.z;  // Z 方向线程索引
```
- **类型**: `uint3`
- **用途**: 确定当前线程在线程块中的位置

### 块维度信息
```cuda
int bdx = blockDim.x;  // X 方向线程数
int bdy = blockDim.y;  // Y 方向线程数
int bdz = blockDim.z;  // Z 方向线程数
```
- **类型**: `dim3`
- **用途**: 获取线程块的维度信息

### 网格维度信息
```cuda
int gdx = gridDim.x;   // X 方向块数
int gdy = gridDim.y;    // Y 方向块数
int gdz = gridDim.z;   // Z 方向块数
```
- **类型**: `dim3`
- **用途**: 获取网格的维度信息

## 5. 内存管理 API

### cudaMalloc(devPtr, size)
```cuda
float* d_data;
cudaMalloc(&d_data, n * sizeof(float));
```
- **参数**: 设备指针的地址, 分配字节数
- **返回**: `cudaError_t` 错误状态码
- **功能**: 在 GPU 上分配全局内存

### cudaFree(devPtr)
```cuda
cudaFree(d_data);
```
- **参数**: 设备内存指针
- **返回**: `cudaError_t` 错误状态码
- **功能**: 释放 GPU 全局内存

### cudaMemcpy(dst, src, count, kind)
```cuda
// 主机 → 设备
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// 设备 → 主机
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
// 设备 → 设备
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDeviceToDevice);
```
- **参数**: 目标指针, 源指针, 字节数, 传输方向
- **返回**: `cudaError_t` 错误状态码
- **传输方向**:
  - `cudaMemcpyHostToDevice`: H2D
  - `cudaMemcpyDeviceToHost`: D2H
  - `cudaMemcpyDeviceToDevice`: D2D

### cudaMemset(devPtr, value, count)
```cuda
cudaMemset(d_data, 0, n * sizeof(float));
```
- **参数**: 设备指针, 填充值（按字节）, 字节数
- **功能**: 将设备内存设为指定值

## 6. 线程同步机制

### __syncthreads()
```cuda
__syncthreads();
```
- **功能**: 等待块内所有线程到达同步点
- **用途**: 确保共享内存数据一致性
- **⚠️ 注意**: 块内所有线程都必须执行到此同步点，否则导致死锁

### __threadfence()
```cuda
__threadfence();
```
- **功能**: 确保当前线程的内存写入对所有线程全局可见
- **用途**: 跨线程块的数据可见性保证

### __threadfence_block()
```cuda
__threadfence_block();
```
- **功能**: 确保当前线程的内存写入对同块线程可见
- **用途**: 块内的内存可见性保证

## 7. 原子操作 API

### atomicAdd(address, val)
```cuda
atomicAdd(&output[0], local_sum);
```
- **功能**: 线程安全的加法操作
- **支持类型**: `int`, `unsigned int`, `float`, `double`（Compute Capability 6.0+）

### atomicMax / atomicMin
```cuda
atomicMax(&max_val, local_max);
atomicMin(&min_val, local_min);
```
- **功能**: 线程安全的最大值/最小值更新
- **支持类型**: `int`, `unsigned int`

### atomicCAS(address, compare, val)
```cuda
int old = atomicCAS(&target, expected, desired);
```
- **功能**: 比较并交换（Compare-And-Swap）
- **返回**: 原始值

### atomicExch(address, val)
```cuda
float old = atomicExch(&target, new_value);
```
- **功能**: 原子交换操作
- **返回**: 原始值

## 8. 数学运算函数

### 标准数学函数
```cuda
float max_val = fmaxf(a, b);      // 最大值
float min_val = fminf(a, b);      // 最小值
float abs_val = fabsf(x);         // 绝对值
float sqrt_val = sqrtf(x);        // 平方根
float rsqrt_val = rsqrtf(x);     // 平方根倒数
float exp_val = expf(x);          // 指数
float log_val = logf(x);          // 自然对数
float pow_val = powf(base, exp);  // 幂运算
float ceil_val = ceilf(x);        // 向上取整
float floor_val = floorf(x);      // 向下取整
```

### 快速数学函数（精度略低但速度更快）
```cuda
float fast_exp = __expf(x);       // 快速指数
float fast_log = __logf(x);       // 快速对数
float fast_sin = __sinf(x);       // 快速正弦
float fast_cos = __cosf(x);       // 快速余弦
float fast_pow = __powf(b, e);    // 快速幂运算
```

### 类型转换
```cuda
float f = __int2float_rn(i);      // int → float（最近舍入）
int i = __float2int_rn(f);        // float → int（最近舍入）
```

## 9. Warp 级操作（Compute Capability 3.0+）

### __shfl_sync / __shfl_down_sync
```cuda
// Warp 内数据交换
float val = __shfl_sync(0xFFFFFFFF, src_val, src_lane);
// Warp 内归约
float val = __shfl_down_sync(0xFFFFFFFF, src_val, offset);
```
- **功能**: Warp 内线程间直接数据交换
- **参数**: mask（参与线程掩码）, 源值, 源 lane / 偏移量

## 10. PyTorch 集成 API

### torch::Tensor 操作
```cuda
// 在 CUDA 源代码中使用 PyTorch 接口
torch::Tensor my_op(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    
    my_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );
    return output;
}
```

### 常用 torch::Tensor 方法
```cuda
input.numel()              // 元素总数
input.size(dim)            // 指定维度大小
input.data_ptr<float>()    // 获取数据指针
torch::zeros_like(input)   // 创建同形状零张量
torch::empty_like(input)   // 创建同形状未初始化张量
```

## 使用建议

1. **边界检查**: 所有数组访问前必须检查边界
2. **原子操作**: 仅在必要时使用，有性能开销
3. **快速数学函数**: 精度要求不高时使用 `__expf` 等加速
4. **共享内存**: 使用前后必须 `__syncthreads()` 同步
5. **错误检查**: 每个 CUDA API 调用后检查返回值
6. **内存对齐**: 确保数据按合适边界对齐（128 字节）
