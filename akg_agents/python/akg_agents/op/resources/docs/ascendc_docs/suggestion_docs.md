# AscendC 开发技巧与优化指南

## 概述

本文档提供 AscendC 算子开发的实用技巧、性能优化方法和常见问题排查指南。

## 开发技巧

### 1. 代码组织技巧

#### 规范流程设计
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

## 性能优化

### 1. 内存访问优化

#### 数据局部性优化
```cpp
// 优化前：频繁的全局内存访问
for (int i = 0; i < size; ++i) {
    result[i] = input[i] + bias[i];  // 每次访问全局内存
}

// 优化后：批量加载到本地内存
AscendC::LocalTensor<float> inputLocal = buffer.AllocTensor<float>();
AscendC::LocalTensor<float> biasLocal = buffer.AllocTensor<float>();
AscendC::LocalTensor<float> resultLocal = buffer.AllocTensor<float>();

AscendC::DataCopy(inputLocal, inputGm, size);
AscendC::DataCopy(biasLocal, biasGm, size);
AscendC::Add(resultLocal, inputLocal, biasLocal);
AscendC::DataCopy(resultGm, resultLocal, size);
```

#### 内存对齐优化
```cpp
// 确保数据按32字节对齐
constexpr uint32_t ALIGNMENT = 32U;

uint32_t AlignSize(uint32_t size, uint32_t typeSize) {
    uint32_t bytes = size * typeSize;
    return (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT / typeSize;
}

// 使用对齐后的尺寸
uint32_t alignedSize = AlignSize(originalSize, sizeof(float));
```

### 2. 计算优化

#### 向量化计算
```cpp
// 使用向量化API替代循环
// 优化前
for (int i = 0; i < size; ++i) {
    output[i] = input[i] * scalar;
}

// 优化后
AscendC::Mul(outputTensor, inputTensor, scalar);
```

### 3. 流水线优化

#### 双缓冲技术
```cpp
class DoubleBufferKernel {
private:
    AscendC::TQue<AscendC::TPosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outputQueue;
    
public:
    __aicore__ inline void Process() {
        // 并行执行数据加载和计算
        CopyIn();   // 加载下一批数据
        Compute();  // 处理当前数据
        CopyOut();  // 输出上一批结果
    }
};
```

## 常见问题排查

### 1. 内存问题

#### 内存泄漏
```cpp
// 问题：忘记释放张量
void ProblematicCode() {
    LocalTensor<float> tensor = buffer.AllocTensor<float>();
    // 使用张量
    // 忘记调用 buffer.FreeTensor(tensor);
}
```

### 2. 性能问题

#### 频繁内存分配
```cpp
// 问题：在循环中频繁分配内存
for (int i = 0; i < iterations; ++i) {
    LocalTensor<float> tensor = buffer.AllocTensor<float>();
    // 使用张量
    buffer.FreeTensor(tensor);
}

// 解决：预分配内存
LocalTensor<float> tensor = buffer.AllocTensor<float>();
for (int i = 0; i < iterations; ++i) {
    // 重用张量
}
buffer.FreeTensor(tensor);
```

#### 低效的数据拷贝
```cpp
// 问题：多次小数据拷贝
for (int i = 0; i < size; ++i) {
    AscendC::DataCopy(&local[i], &global[i], 1);
}

// 解决：批量拷贝
AscendC::DataCopy(localTensor, globalTensor, size);
```

### 3. 数据类型使用问题

#### 在AscendC中，对输入的结果出现NaN的情况，需要检查任务的输入数据类型与AscendC所采用的类型是否一致。
 - 例如：float32类型的输入数据，在AscendC中需要使用float类型进行计算，避免出现NaN的情况。


## 内核算子生成规范

### 任务描述
AscendC的实现主要包含三个部分，分别是host侧Tiling的实现部分，device侧kernel实现部分以及PyBind模块与算子核函数进行绑定并封装为Python模块的内容，分别用这三种字段做区分：host_tiling_src，kernel_src，python_bind_src。
【注意】
请严格按照以下的格式输出！

```

host_tiling_src="""
#include "tiling/tiling_api.h"
uint8_t *GetTilingBuf()
{
    //AscendC Host侧Tiling实现代码段，对于Elemwise类的算子，不需要复杂的Tiling,直接返回nullptr即可，在kernel中进行切分即可。
}
"""

kernel_src="""
#include "kernel_operator.h"
class KernelOpName {
    //AscendC代码段
};


extern "C" __global__ __aicore__ void op_name_kernel()
{
    KernelOpName op;
    op.Init();
    op.Process()
}
"""

python_bind_src="""
#include <pybind11/pybind11.h>
#include <torch/extension.h>

//内核调用操作...

//绑定操作
PYBIND11_MODULE(op_name_kernel, m)
{
    m.doc() = "op_name_kernel pybind11 interfaces"; // optional module docstring
    m.def("run_op_name_kernel", &namespace::run_op_name_kernel, "");
}
"""
请注意上述的上述的PyBind_Module的使用方式，不要出现模块导入与调用的错误。
```
