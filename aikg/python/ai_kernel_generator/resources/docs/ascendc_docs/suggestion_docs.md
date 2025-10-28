# AscendC 开发技巧与优化指南

## 概述

本文档提供 AscendC 算子开发的实用技巧、性能优化方法和常见问题排查指南。

## 开发技巧

### 1. 代码组织技巧

#### 模板化设计
```cpp
// 使用模板提高代码复用性
template <typename T, bool isTransA, bool isTransB>
class MatmulKernel {
public:
    __aicore__ inline void Process() {
        // 通用处理逻辑
    }
};

// 特化版本
template <>
class MatmulKernel<float, true, false> {
    // 特定优化实现
};
```

### 2. 错误处理技巧

#### 异常安全设计
```cpp
class SafeKernel {
private:
    bool initialized = false;
    AscendC::TPipe pipe;
    
public:
    __aicore__ inline bool Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling) {
        try {
            // 初始化逻辑
            pipe.InitBuffer(buffer, size, bufferSize);
            initialized = true;
            return true;
        } catch (...) {
            initialized = false;
            return false;
        }
    }
    
    __aicore__ inline void Process() {
        ASCENDC_ASSERT(initialized, {
            KERNEL_LOG(KERNEL_ERROR, "Kernel not initialized");
        });
        
        // 处理逻辑
    }
};
```

#### 参数验证
```cpp
class ParameterValidator {
public:
    static bool ValidateMatmulParams(uint32_t M, uint32_t N, uint32_t K) {
        if (M == 0 || N == 0 || K == 0) {
            KERNEL_LOG(KERNEL_ERROR, "Invalid matrix dimensions");
            return false;
        }
        
        if (M > MAX_MATRIX_SIZE || N > MAX_MATRIX_SIZE || K > MAX_MATRIX_SIZE) {
            KERNEL_LOG(KERNEL_ERROR, "Matrix size exceeds limit");
            return false;
        }
        
        return true;
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

#### 融合操作
```cpp
// 融合多个操作减少内存访问
class FusedKernel {
public:
    __aicore__ inline void Process() {
        // 融合：Add + Relu + Mul
        AscendC::Add(tmpTensor, inputA, inputB);
        AscendC::Relu(tmpTensor, tmpTensor);
        AscendC::Mul(outputTensor, tmpTensor, scale);
    }
};
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

#### 异步处理
```cpp
class AsyncKernel {
public:
    __aicore__ inline void Process() {
        // 启动异步数据加载
        StartAsyncCopy();
        
        // 处理当前数据
        Compute();
        
        // 等待异步操作完成
        WaitForAsyncCopy();
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

// 解决：使用RAII模式
class TensorWrapper {
private:
    AscendC::LocalTensor<float> tensor;
    AscendC::TBuf<AscendC::TPosition::UB>& buffer;
    
public:
    TensorWrapper(AscendC::TBuf<AscendC::TPosition::UB>& buf) 
        : buffer(buf) {
        tensor = buffer.AllocTensor<float>();
    }
    
    ~TensorWrapper() {
        buffer.FreeTensor(tensor);
    }
    
    AscendC::LocalTensor<float>& get() { return tensor; }
};
```

#### 内存对齐问题
```cpp
// 问题：数据未对齐导致性能下降
uint32_t size = 100;  // 不是32字节对齐

// 解决：确保对齐
uint32_t alignedSize = (size + 31) / 32 * 32;
```

### 2. 计算问题

#### 数值精度问题
```cpp
// 问题：float16精度损失
half a = 0.1f;
half b = 0.2f;
half result = a + b;  // 可能不精确

// 解决：使用float进行计算
float a_f = static_cast<float>(a);
float b_f = static_cast<float>(b);
float result_f = a_f + b_f;
half result = static_cast<half>(result_f);
```

#### 溢出问题
```cpp
// 问题：整数溢出
int32_t a = INT32_MAX;
int32_t b = 1;
int32_t result = a + b;  // 溢出

// 解决：使用更大的数据类型
int64_t result = static_cast<int64_t>(a) + static_cast<int64_t>(b);
```

### 3. 性能问题

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
    //AscendC Host侧Tiling实现代码段
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
