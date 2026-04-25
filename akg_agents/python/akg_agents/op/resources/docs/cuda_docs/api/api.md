# CUDA C 编程接口指南

本文介绍 CUDA C 编程中的核心接口和函数。

## 1. 内核函数定义与调用

### 内核启动语法
```cuda
kernel_name<<<grid_size, block_size>>>(参数列表);
```
- **grid_size**: 网格大小配置
- **block_size**: 线程块大小配置
- **参数列表**: 传递给内核的参数

## 2. 线程和块索引系统

### 块索引变量
```cuda
int block_x = blockIdx.x;  // X 方向块索引
int block_y = blockIdx.y;  // Y 方向块索引
int block_z = blockIdx.z;  // Z 方向块索引
```
- **功能**: 获取当前线程块在网格中的位置
- **应用**: 用于确定数据处理的起始位置

### 线程索引变量
```cuda
int thread_x = threadIdx.x;  // X 方向线程索引
int thread_y = threadIdx.y;  // Y 方向线程索引
int thread_z = threadIdx.z;  // Z 方向线程索引
```
- **功能**: 获取当前线程在线程块中的位置
- **应用**: 用于计算线程在块内的相对位置

### 块维度信息
```cuda
int block_width = blockDim.x;   // X 方向线程数
int block_height = blockDim.y;  // Y 方向线程数
int block_depth = blockDim.z;   // Z 方向线程数
```
- **功能**: 获取线程块的维度信息
- **应用**: 用于边界检查和索引计算

### 网格维度信息
```cuda
int grid_width = gridDim.x;   // X 方向块数
int grid_height = gridDim.y;  // Y 方向块数
int grid_depth = gridDim.z;   // Z 方向块数
```
- **功能**: 获取网格的维度信息
- **应用**: 用于计算总的处理范围

## 3. 全局索引计算方法

### 一维数据处理
```cuda
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```
- **用途**: 计算线程的全局一维索引
- **适用**: 处理一维数组或向量

### 二维数据处理
```cuda
int row_index = blockIdx.y * blockDim.y + threadIdx.y;
int col_index = blockIdx.x * blockDim.x + threadIdx.x;
```
- **用途**: 计算二维数据访问的行列索引
- **适用**: 处理矩阵或图像数据

### 三维数据处理
```cuda
int x_coord = blockIdx.x * blockDim.x + threadIdx.x;
int y_coord = blockIdx.y * blockDim.y + threadIdx.y;
int z_coord = blockIdx.z * blockDim.z + threadIdx.z;
```
- **用途**: 计算三维数据访问的坐标
- **适用**: 处理体积数据或张量

### 网格配置示例
```cuda
// 处理 1024x1024 图像
dim3 block_size(16, 16);
dim3 grid_size(64, 64);

// 处理批量数据
dim3 block_size(8, 8, 8);
dim3 grid_size(4, 4, 4);

// 启动内核
process_data<<<grid_size, block_size>>>(data_ptr, width, height);
```

## 4. 内存管理接口

### 设备内存分配
```cuda
float* device_memory;
cudaMalloc(&device_memory, data_size * sizeof(float));
```
- **参数**: 设备指针地址, 分配字节数
- **返回值**: 错误状态码
- **功能**: 在 GPU 上分配内存空间

### 设备内存释放
```cuda
cudaFree(device_memory);
```
- **参数**: 设备内存指针
- **返回值**: 错误状态码
- **功能**: 释放 GPU 内存空间

### 内存数据传输
```cuda
cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
```
- **参数**: 目标指针, 源指针, 字节数, 传输方向
- **返回值**: 错误状态码
- **功能**: 在主机和设备间传输数据

## 5. 数学运算函数

### 最大值函数
```cuda
float max_value = fmaxf(value_a, value_b);
```
- **参数**: 两个浮点数值
- **返回值**: 较大的数值
- **用途**: 比较并返回较大值

### 绝对值函数
```cuda
float abs_value = fabsf(input_value);
```
- **参数**: 浮点数值
- **返回值**: 绝对值
- **用途**: 计算数值的绝对值

## 6. 线程同步机制

### 块内同步
```cuda
__syncthreads();
```
- **功能**: 等待块内所有线程到达同步点
- **用途**: 确保数据一致性，协调线程执行

### 内存屏障
```cuda
__threadfence();
```
- **功能**: 确保内存操作的全局可见性
- **用途**: 保证内存写入对其他线程可见

## 7. 函数修饰符

### 内核函数修饰符
```cuda
__global__ void kernel_function(参数列表);
```
- **功能**: 标记为可在 GPU 上执行的内核函数
- **调用**: 只能从主机代码调用

### 设备函数修饰符
```cuda
__device__ float device_function(参数列表);
```
- **功能**: 标记为在 GPU 上执行的设备函数
- **调用**: 只能从其他设备函数调用

### 主机函数修饰符
```cuda
__host__ void host_function(参数列表);
```
- **功能**: 标记为在 CPU 上执行的函数
- **调用**: 只能从主机代码调用

### 内存类型修饰符
```cuda
__shared__ float shared_memory[256];  // 共享内存
__constant__ float constant_data[64]; // 常量内存
```
- **shared**: 线程块内共享的内存
- **constant**: 只读的常量内存
```