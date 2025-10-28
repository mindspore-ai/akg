# AscendC API 参考文档

## 概述

AscendC 是华为昇腾 AI 处理器的高性能编程语言，提供了丰富的 API 用于开发 AI 算子内核。本文档整理了 AscendC 中最常用的 API 用于算子开发。

## 核心数据类型

### 张量类型
- `AscendC::GlobalTensor<T>` - 全局内存张量
- `AscendC::LocalTensor<T>` - 本地内存张量
- `AscendC::Tensor<T>` - 通用张量类型

### 内存位置
- `AscendC::TPosition::GM` - 全局内存
- `AscendC::TPosition::UB` - 统一缓冲区
- `AscendC::TPosition::VECIN` - 向量输入队列
- `AscendC::TPosition::VECOUT` - 向量输出队列
- `AscendC::TPosition::VECCALC` - 向量计算缓冲区

## 数学运算 API

### 基础运算
```cpp
// 加法
AscendC::Add(dst, src0, src1);

// 乘法
AscendC::Mul(dst, src0, src1);

// 取最大值
AscendC::Max(dst, src0, src1);


### 归约运算
```cpp
// 求和
AscendC::Sum(dst, src, tmpBuffer, params);

// 求平均值
AscendC::Mean(dst, src, tmpBuffer, params);

// 求最大值
AscendC::ReduceMax(dst, src, tmpBuffer, params);

// 求最小值
AscendC::ReduceMin(dst, src, tmpBuffer, params);
```

### 激活函数
```cpp
// ReLU
AscendC::Relu(dst, src);

// Sigmoid
AscendC::Sigmoid(dst, src);

// Softmax
AscendC::SoftMax(dst, src, tmpBuffer, params);
```

## 矩阵运算 API

### 矩阵乘法
```cpp
// 创建 Matmul 对象
AscendC::Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC, MatmulTypeBias, CFG> matmulObj;

// 设置输入矩阵
matmulObj.SetTensorA(aTensor, isTransA);
matmulObj.SetTensorB(bTensor, isTransB);

// 设置偏置
matmulObj.SetBias(biasTensor);

// 执行矩阵乘法
matmulObj.IterateAll(cTensor);
matmulObj.End();
```

## 数据操作 API

### 内存拷贝
```cpp
// 全局内存到本地内存
AscendC::DataCopy(localTensor, globalTensor, length);

// 本地内存到全局内存
AscendC::DataCopy(globalTensor, localTensor, length);

// 本地内存间拷贝
AscendC::DataCopy(dstLocal, srcLocal, length);
```

### 数据初始化
```cpp
// 用标量填充张量
AscendC::Duplicate(tensor, scalar, length);

// 用零填充
AscendC::Memset(tensor, 0, length);

// 用指定值填充
AscendC::Memset(tensor, value, length);
```

## 管道和队列 API

### 管道管理
```cpp
// 创建管道
AscendC::TPipe pipe;

// 初始化缓冲区
pipe.InitBuffer(buffer, queueSize, bufferSize);

// 注册 Matmul 对象
REGIST_MATMUL_OBJ(&pipe, workspacePtr, matmulObj, &tiling);
```

### 队列操作
```cpp
// 创建队列
AscendC::TQue<TPosition, queueSize> queue;

// 分配张量
LocalTensor<T> tensor = queue.AllocTensor<T>();

// 入队
queue.EnQue(tensor);

// 出队
LocalTensor<T> tensor = queue.DeQue<T>();

// 释放张量
queue.FreeTensor(tensor);
```

## 缓冲区管理 API

### 缓冲区操作
```cpp
// 创建缓冲区
AscendC::TBuf<TPosition> buffer;

// 分配张量
LocalTensor<T> tensor = buffer.AllocTensor<T>();

// 释放张量
buffer.FreeTensor(tensor);
```

## 核函数开发 API

### 核函数入口
```cpp
extern "C" __global__ __aicore__ void kernel_name(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    // 核函数实现
}
```

### 核函数调用
```cpp
void kernel_name_do(uint32_t blockDim, void* stream,
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    kernel_name<<<blockDim, nullptr, stream>>>(input0, input1, output, workspace, tiling);
}
```

## 数据类型支持

### 基础数据类型
- `half` - 16位浮点数
- `float` - 32位浮点数
- `int8_t` - 8位整数
- `int16_t` - 16位整数
- `int32_t` - 32位整数
- `uint8_t` - 8位无符号整数
- `uint16_t` - 16位无符号整数
- `uint32_t` - 32位无符号整数

### 张量格式
- `CubeFormat::ND` - N维张量格式
- `CubeFormat::FRACTAL_NZ` - 分形NZ格式
- `CubeFormat::FRACTAL_Z` - 分形Z格式