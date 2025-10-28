# AscendC 核心概念与编程模式

## 概述

本文档介绍 AscendC 的核心概念和常用编程模式。

## 核心架构

### 1. 内存层次结构

AscendC 采用分层内存架构：

**内存类型说明：**
- **Global Memory (GM)**: 大容量存储，访问延迟较高
- **Unified Buffer (UB)**: 片上高速缓存，容量小但速度快
- **Vector Queue (VEC)**: 用于数据流水线传输

### 2. 计算核心

AscendC 支持多种计算核心：
- **AIC (AI Core)**: 主要计算核心，支持矩阵运算
- **AIV (AI Vector)**: 向量计算核心，支持向量运算
- **CPU**: 通用计算核心

## 核心概念

### 1. 张量 (Tensor)

张量是 AscendC 中的基本数据结构：

```cpp
// 全局张量 - 存储在全局内存
AscendC::GlobalTensor<float> globalTensor;

// 本地张量 - 存储在统一缓冲区
AscendC::LocalTensor<float> localTensor;
```

**张量操作模式：**
```cpp
// 1. 设置缓冲区
globalTensor.SetGlobalBuffer((__gm__ float*)ptr, size);

// 2. 数据拷贝
AscendC::DataCopy(localTensor, globalTensor, length);

// 3. 计算操作
AscendC::Add(resultTensor, tensorA, tensorB);

// 4. 结果写回
AscendC::DataCopy(globalTensor, resultTensor, length);
```

### 2. 管道 (Pipe)

管道是 AscendC 的核心抽象，用于管理数据流和计算资源：

```cpp
AscendC::TPipe pipe;

// 初始化缓冲区
pipe.InitBuffer(buffer, queueSize, bufferSize);

// 注册计算对象
REGIST_MATMUL_OBJ(&pipe, workspacePtr, matmulObj, &tiling);
```

### 3. 队列 (Queue)

队列用于实现数据流水线：

```cpp
// 输入队列
AscendC::TQue<AscendC::TPosition::VECIN, queueSize> inQueue;

// 输出队列  
AscendC::TQue<AscendC::TPosition::VECOUT, queueSize> outQueue;

// 队列操作
LocalTensor<T> tensor = inQueue.AllocTensor<T>();
inQueue.EnQue(tensor);
LocalTensor<T> result = outQueue.DeQue<T>();
```

### 4. 缓冲区 (Buffer)

缓冲区用于临时数据存储：

```cpp
AscendC::TBuf<AscendC::TPosition::UB> tmpBuf;

// 分配张量
LocalTensor<T> tensor = tmpBuf.AllocTensor<T>();

// 释放张量
tmpBuf.FreeTensor(tensor);
```

## 编程模式

### 1. Copy-Compute-Copy 模式

这是 AscendC 中最基本的编程模式：

```cpp
class BasicKernel {
public:
    __aicore__ inline void Process() {
        CopyIn();    // 数据从全局内存拷贝到本地内存
        Compute();   // 在本地内存中进行计算
        CopyOut();   // 结果从本地内存拷贝回全局内存
    }

private:
    __aicore__ inline void CopyIn() {
        AscendC::DataCopy(localTensor, globalTensor, length);
    }
    
    __aicore__ inline void Compute() {
        AscendC::Add(resultTensor, tensorA, tensorB);
    }
    
    __aicore__ inline void CopyOut() {
        AscendC::DataCopy(globalTensor, resultTensor, length);
    }
};
```

### 2. 流水线模式

使用队列实现数据流水线：

```cpp
class PipelineKernel {
public:
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn() {
        LocalTensor<T> input = inQueue.AllocTensor<T>();
        AscendC::DataCopy(input, inputGm, length);
        inQueue.EnQue(input);
    }
    
    __aicore__ inline void Compute() {
        LocalTensor<T> input = inQueue.DeQue<T>();
        LocalTensor<T> output = outQueue.AllocTensor<T>();
        
        // 执行计算
        AscendC::Add(output, input, bias);
        
        outQueue.EnQue(output);
        inQueue.FreeTensor(input);
    }
    
    __aicore__ inline void CopyOut() {
        LocalTensor<T> output = outQueue.DeQue<T>();
        AscendC::DataCopy(outputGm, output, length);
        outQueue.FreeTensor(output);
    }
};
```

### 3. 矩阵乘法模式

使用高阶 API 实现矩阵乘法：

```cpp
template <typename aType, typename bType, typename cType, typename biasType>
class MatmulKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, const TCubeTiling& tiling) {
        // 设置全局张量
        aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType*>(a), tiling.M * tiling.Ka);
        bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType*>(b), tiling.Kb * tiling.N);
        cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType*>(c), tiling.M * tiling.N);
        
        // 计算偏移
        CalcOffset(AscendC::GetBlockIdx(), offsetA, offsetB, offsetC, offsetBias);
        
        // 设置张量偏移
        aGlobal = aGlobal[offsetA];
        bGlobal = bGlobal[offsetB];
        cGlobal = cGlobal[offsetC];
    }
    
    __aicore__ inline void Process(AscendC::TPipe* pipe) {
        // 设置输入矩阵
        matmulObj.SetTensorA(aGlobal, isAtrans);
        matmulObj.SetTensorB(bGlobal, isBtrans);
        
        // 设置偏置
        if (tiling.isBias) {
            matmulObj.SetBias(biasGlobal);
        }
        
        // 执行矩阵乘法
        matmulObj.IterateAll(cGlobal);
        matmulObj.End();
    }
};
```

### 4. 归约运算模式

实现归约运算（如求和、求最大值等）：

```cpp
template <typename T>
class ReduceKernel {
public:
    __aicore__ inline void Process() {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void Compute() {
        LocalTensor<T> input = inQueue.DeQue<T>();
        LocalTensor<T> output = outQueue.AllocTensor<T>();
        LocalTensor<uint8_t> tmpBuffer = tmpBuf.AllocTensor<uint8_t>();
        
        // 初始化输出
        T scalar(0);
        AscendC::Duplicate<T>(output, scalar, outLength);
        
        // 执行归约运算
        AscendC::Sum(output, input, tmpBuffer, params);
        
        outQueue.EnQue<T>(output);
        inQueue.FreeTensor(input);
        tmpBuf.FreeTensor(tmpBuffer);
    }
};
```

## 内存管理策略

### 1. 内存对齐

AscendC 要求数据按32字节对齐以获得最佳性能：

```cpp
constexpr uint32_t PADDING_BYTE = 32U;

auto paddingFunc = [](const uint32_t n, const uint32_t typeSize) -> uint32_t {
    if (typeSize == 0) return 0;
    return (n * typeSize + PADDING_BYTE - 1U) / PADDING_BYTE * PADDING_BYTE / typeSize;
};
```

### 2. 工作空间管理

```cpp
// 获取系统工作空间
void* workspacePtr = GetSysWorkSpacePtr();

// 设置本地工作空间
matmulObj.SetLocalWorkspace(localWorkspace);
```

### 3. 内存生命周期

```cpp
// 正确的内存管理流程
{
    LocalTensor<T> tensor = buffer.AllocTensor<T>();
    // 使用张量
    // ...
    buffer.FreeTensor(tensor);  // 必须释放
}
```

## 核函数开发模式

### 1. 核函数入口

```cpp
extern "C" __global__ __aicore__ void kernel_name(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, 
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 平台检测
    if ASCEND_IS_AIC {
        return;
    }
    
    // 创建管道
    AscendC::TPipe pipe;
    
    // 创建核函数对象
    KernelClass kernel;
    
    // 初始化
    kernel.Init(input0, input1, output, tiling, &pipe);
    
    // 执行
    kernel.Process();
}
```

### 2. 核函数调用

```cpp
void kernel_name_do(uint32_t blockDim, void* stream,
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, 
    GM_ADDR workspace, GM_ADDR tiling)
{
    kernel_name<<<blockDim, nullptr, stream>>>(
        input0, input1, output, workspace, tiling);
}
```
## PyBind11 开发指南
### Python模块绑定
AscendC代码通过PyBind11与Python交互，实现Python模块调用

### PyBind11基本结构
``` cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "aclrtlaunch_add_custom.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace my_kernel {
at::Tensor run_kernel_custom(const at::Tensor &x, const at::Tensor &y)
{
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor z = at::empty_like(x);
    uint32_t blockDim = 8;
    uint32_t totalLength = 1;
    for (uint32_t size : x.sizes()) {
        totalLength *= size;
    }
    ACLRT_LAUNCH_KERNEL(kernel_name)
    // 其他代码
}
} // namespace my_kernel

PYBIND11_MODULE(kernel_name, m)
{
    m.doc() = "add_custom pybind11 interfaces"; // optional module docstring
    m.def("run_kernel_custom", &my_kernel::run_kernel_custom, "");
}
```
### 关键组件说明
- **头文件包含**：pybind11、torch扩展、ACL启动、NPU流
- **命名空间**：避免与函数名冲突
- **张量处理**：使用at::Tensor进行张量操作
- **流管理**：获取当前NPU流进行内核启动
- **模块定义**：PYBIND11_MODULE宏定义Python模块

