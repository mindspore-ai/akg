# AscendC 核心概念与编程模式

## 概述

本文档介绍 AscendC 的核心概念和常用编程模式。

## 核心架构

### 1. 内存层次结构

AscendC 采用分层内存架构：

**内存类型说明：**
- **Global Memory (GM)**: 大容量存储，访问延迟较高
- **Vector Queue (VEC)**: 用于数据流水线传输

### 2. 计算核心

AscendC 支持多种计算核心：
- **AIC (AI Core)**: 主要计算核心，支持矩阵运算
- **AIV (AI Vector)**: 向量计算核心，支持向量运算

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
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
    uint32_t byteLen = 1024;
    pipe.InitBuffer(calcBuf, byteLen);
    // 从calcBuf获取Tensor,Tensor为pipe分配的所有内存大小，为1024字节
    AscendC::LocalTensor<int32_t> tempTensor1 = calcBuf.Get<int32_t>();
    // 从calcBuf获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
    AscendC::LocalTensor<int32_t> tempTensor2 = calcBuf.Get<int32_t>(128);
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

使用队列实现加法数据流水线：

```cpp
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
class KernelAdd {
public:
    __aicore__ inline KernelOps() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = 8;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
```
### 3. 核函数入口
```cpp
   extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    op.Process();
}
```
### 3. 核函数调用

```cpp
    ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, acl_stream,
        cast<void*>(x.storage().data()), cast<void*>(y.storage().data()), cast<void*>(z.storage().data()), totalLength);
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
    ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, acl_stream,
        x.data_ptr(), y.data_ptr(), z.data_ptr(), totalLength);
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
- **头文件包含**：pybind11、torch扩展、算子头文件、流管理头文件
- **命名空间**：避免与函数名冲突
- **张量处理**：使用at::Tensor进行张量操作
- **流管理**：获取当前NPU流进行内核启动
- **模块定义**：PYBIND11_MODULE宏定义Python模块

