# AscendC API 参考文档

## 概述

AscendC 是华为昇腾 AI 处理器的高性能编程语言，提供了丰富的 API 用于开发 AI 算子内核。Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。本文档整理了 AscendC 中最常用的 API 用于算子开发。为方便开发者使用，Ascend C基础API和高阶API均支持通过包含kernel_operator.h文件来调用相应接口。如无特殊说明，包含该头文件（#include "kernel_operator.h"）即可满足接口调用需求。若API文档中有特殊说明，则应遵循API的具体说明。

## 核心数据类型
### 基础数据类型
 - int32_t：32位有符号整数
 - float：单精度浮点数
 - half：半精度浮点数
 - int64_t：64位有符号整数
 - int8_t：8位有符号整数
 - uint8_t：8位无符号整数
 - uint32_t：32位无符号整数
 - uint64_t：64位无符号整数 
 - bfloat16_t：16位半精度浮点数

### 张量类型

#### `AscendC::LocalTensor<T>` - 本地内存张量，存储在统一缓冲区中，LocalTensor用于存放AI Core中Local Memory（内部存储）的数据，支持逻辑位置TPosition为VECIN、VECOUT等。
 - 说明：类型T可以支持基础数据类型，如int32_t、float、half等。
 - 使用示例：AscendC::LocalTensor<half> tensor1 = que1.DeQue<half>();
#### `AscendC::GlobalTensor<T>` - 全局内存张量，存储在设备全局内存中， GlobalTensor用来存放Global Memory（外部存储）的全局数据。
 - 说明：类型T可以支持基础数据类型，如int32_t、float、half等。
 - 使用示例：AscendC::GlobalTensor<int32_t> inputGlobal;
 - 常用操作：__aicore__ inline void SetGlobalBuffer(__gm__ PrimType* buffer, uint64_t bufferSize)，SetGlobalBuffer：设置全局内存张量的起始地址和大小。
#### 使用示例：
```cpp
// 创建全局内存张量
void Init(__gm__ uint8_t *src_gm, __gm__ uint8_t *dst_gm)
{
    uint64_t dataSize = 256; //设置input_global的大小为256
    AscendC::GlobalTensor<float> inputGlobal; // 类型为float
    inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(src_gm), dataSize); // 设置源操作数在Global Memory上的起始地址为src_gm，所占外部存储的大小为256个float
    AscendC::LocalTensor<float> inputLocal = inQueueX.AllocTensor<float>();    
    AscendC::DataCopy(inputLocal, inputGlobal, dataSize); // 将Global Memory上的inputGlobal拷贝到Local Memory的inputLocal上
}
```
## 基础 API

### 基础算术运算
#### Sqrt
 - 函数原型：__aicore__ inline void Sqrt(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)，其中T支持的数据类型有float或half。
 - 使用示例：AscendC::Sqrt(dstLocal, srcLocal, count); // 对srcLocal中的前count个元素计算平方根，结果存储在dstLocal中

#### Relu
 - 函数原型：__aicore__ inline void Relu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)，其中T支持的数据类型有float、half、int32_t。
 - 使用示例：AscendC::Relu(dstLocal, srcLocal, count); // 对srcLocal中的前count个元素计算ReLU函数，结果存储在dstLocal中

#### Add
 - 函数原型：__aicore__ inline void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)，或者dst = src0 + src1，其中T支持的数据类型有float、half、int32_t。
 - 使用示例：AscendC::Add(dstLocal, srcLocal0, srcLocal1, count); // 对srcLocal0和srcLocal1中的前count个元素进行加法运算，结果存储在dstLocal中。或者dstLocal = srcLocal0 + srcLocal1。直接计算srcLocal0和srcLocal1的元素级加法。

#### Mul
 - 函数原型：__aicore__ inline void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)，或者dst = src0 * src1，其中T支持的数据类型有float、half、int32_t。
 - 使用示例：AscendC::Mul(dstLocal, srcLocal0, srcLocal1, count); // 对srcLocal0和srcLocal1中的前count个元素进行乘法运算，结果存储在dstLocal中。或者dstLocal = srcLocal0 * srcLocal1。直接计算srcLocal0和srcLocal1的元素级乘法。

#### Div
 - 函数原型：__aicore__ inline void Div(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)，或者dst = src0 / src1，其中T支持的数据类型有float、half。
 - 使用示例：AscendC::Div(dstLocal, srcLocal0, srcLocal1, count); // 对srcLocal0和srcLocal1中的前count个元素进行除法运算，结果存储在dstLocal中。或者dstLocal = srcLocal0 / srcLocal1。直接计算srcLocal0和srcLocal1的元素级除法。

#### Max
 - 函数原型：__aicore__ inline void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)，其中T支持的数据类型有float、half、int32_t。
 - 使用示例：AscendC::Max(dstLocal, srcLocal0, srcLocal1, count); // 对srcLocal0和srcLocal1中的前count个元素取最大值，结果存储在dstLocal中。

#### LeakyRelu
 - 函数原型：__aicore__ inline void LeakyRelu(const LocalTensor<T>& dst, const LocalTensor<T>& src, const int32_t& count)，其中T支持的数据类型有float、half。
 - 使用示例：AscendC::LeakyRelu(dstLocal, srcLocal, count); // 对srcLocal中的前count个元素计算LeakyReLU函数，结果存储在dstLocal中。

### 数据类型转换
#### Cast（精度转换：根据源操作数和目的操作数Tensor的数据类型进行精度转换。）
 - 函数原型：__aicore__ inline void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& round_mode, const uint32_t count)，其中T和U支持的数据类型有float、half、int32_t、uint32_t、bfloat16_t等。其中RoundMode为精度转换处理模式，RoundMode为枚举类型，用以控制精度转换处理模式，具体定义为：
``` cpp
enum class RoundMode {
    CAST_NONE = 0,  // 在转换有精度损失时表示CAST_RINT模式，不涉及精度损失时表示不舍入
    CAST_RINT,      // rint，四舍六入五成双舍入
    CAST_FLOOR,     // floor，向负无穷舍入
    CAST_CEIL,      // ceil，向正无穷舍入
    CAST_ROUND,     // round，四舍五入舍入
    CAST_TRUNC,     // trunc，向零舍入
    CAST_ODD,       // Von Neumann rounding，最近邻奇数舍入   
};
```
 - 使用示例：AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_CEIL, count); // 对srcLocal中的前count个元素进行精度转换，结果存储在dstLocal中。

### 数据搬运
#### DataCopy（DataCopy系列接口提供全面的数据搬运功能，支持多种数据搬运场景，并可在搬运过程中实现随路格式转换和量化激活等操作。该接口支持Local Memory与Global Memory之间的数据搬运，以及Local Memory内部的数据搬运。）
 - 基础的数据搬运能力，支持连续和非连续的数据搬运。
 - 函数原型举例：
    // 连续搬运
    template <typename T>
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const uint32_t count)

    // 同时支持非连续搬运和连续搬运
    template <typename T>
    __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyParams& repeatParams)
 - 使用示例：
 ``` cpp
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
    pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
    AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
    AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    // 使用传入count参数的搬运接口，完成连续搬运
    AscendC::DataCopy(srcLocal, srcGlobal, 512);
    AscendC::DataCopy(dstLocal , srcLocal, 512);
    AscendC::DataCopy(dstGlobal, dstLocal, 512);
    // 使用传入DataCopyParams参数的搬运接口，支持连续和非连续搬运
    // DataCopyParams intriParams;
    // AscendC::DataCopy(srcLocal, srcGlobal, intriParams);
    // AscendC::DataCopy(dstLocal , srcLocal, intriParams);
    // AscendC::DataCopy(dstGlobal, dstLocal, intriParams);
 ```
### 内存管理
#### TPosition （Ascend C管理不同层级的物理内存时，用一种抽象的逻辑位置（TPosition）来表达各级别的存储，代替了片上物理存储的概念，达到隐藏硬件架构的目的。）
```cpp
enum class TPosition : uint8_t {
    GM,
    A1,
    A2,
    B1,
    B2,
    ...
    VECIN,
    VECOUT,
    VECCALC,
    ...
};
```
 - GM：Global Memory，对应AI Core的外部存储。
 - VECIN：Vector Input，对应AI Core的矢量输入寄存器。
 - VECOUT：Vector Output，对应AI Core的矢量输出寄存器。
 - VECCALC：Vector Calculate，对应AI Core的矢量计算寄存器。
 - A1：用于矩阵计算，存放整块A矩阵，可类比CPU多级缓存中的二级缓存。
 - B1：用于矩阵计算，存放整块B矩阵，可类比CPU多级缓存中的二级缓存。
 - C1：用于矩阵计算，存放整块Bias矩阵，可类比CPU多级缓存中的二级缓存。
 - A2：用于矩阵计算，存放切分后的小块A矩阵，可类比CPU多级缓存中的一级缓存。
 - B2：用于矩阵计算，存放切分后的小块B矩阵，可类比CPU多级缓存中的一级缓存。
 - C2：用于矩阵计算，存放切分后的小块Bias矩阵，可类比CPU多级缓存中的一级缓存。

#### TPipe（用于统一管理Device端内存等资源，一个Kernel函数必须且只能初始化一个TPipe对象）
 - 构造函数：__aicore__ inline TPipe()
 - 调用示例
 ``` cpp
    class KernelExample {
        public:
            __aicore__ inline KernelExample() {}
            __aicore__ inline void Init(..., TPipe* pipeIn)
            {
                ...
                pipe = pipeIn;
                pipe->InitBuffer(xxxBuf, BUFFER_NUM, xxxSize);
                ...
            }
        private:
            ...
            TPipe* pipe;
            ...
    };
    extern "C" __global__ __aicore__ void example_kernel(...) {
        ...
        TPipe pipe;
        KernelExample<float> op;
        op.Init(..., &pipe);
        ...
    }
 ```
 - 常用API
    - Init：用于内存和同步流水事件EventID的初始化。
    - InitBuffer：用于为TQue等队列和TBuf分配内存：
    ``` cpp
        // 为TQue分配内存，分配内存块数为2，每块大小为128字节
        AscendC::TPipe pipe; // Pipe内存管理对象
        AscendC::TQue<AscendC::TPosition::VECOUT, 2> que; // 输出数据队列管理对象，TPosition为VECOUT
        uint8_t num = 2;
        uint32_t len = 128;
        pipe.InitBuffer(que, num, len);
    ```
    - Destroy：用于重复申请释放tpipe，创建tpipe对象后，可调用Destroy手动释放资源。

#### TQue（流水任务之间通过队列（Queue）完成任务间通信和同步。TQue是用来执行队列相关操作、管理相关资源的数据结构。）
常用API：
 - EnQue：入队，将数据写入队列。
 - DeQue：出队，从队列读取数据。
 - AllocTensor：分配张量，从队列中分配一段内存用于存储数据。
 - FreeTensor：释放张量，将分配的内存返回给队列。
使用示例：
```cpp
    // 入队示例
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    pipe.InitBuffer(inQueueSrc, 1, 512 * sizeof(half));
    pipe.InitBuffer(outQueueDst, 1, 512 * sizeof(half));
    AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
    AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    // 入队
    inQueueSrc.EnQue(srcLocal);
    // 出队
    outQueueDst.DeQue(dstLocal);
```

#### TBuf（使用Ascend C编程的过程中，可能会用到一些临时变量。这些临时变量占用的内存可以使用TBuf数据结构来管理，存储位置通过模板参数来设置，可以设置为不同的TPosition逻辑位置。）
常用API：
 - Get：从TBuf上获取指定长度的Tensor，或者获取全部长度的Tensor。
使用示例：
```cpp
    // 为TBuf初始化分配内存，分配内存长度为1024字节
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf; // 模板参数为TPosition中的VECCALC类型
    uint32_t byteLen = 1024;
    pipe.InitBuffer(calcBuf, byteLen);
    // 从calcBuf获取Tensor,Tensor为pipe分配的所有内存大小，为1024字节
    AscendC::LocalTensor<int32_t> tempTensor1 = calcBuf.Get<int32_t>();
    // 从calcBuf获取Tensor,Tensor为128个int32_t类型元素的内存大小，为512字节
    AscendC::LocalTensor<int32_t> tempTensor2 = calcBuf.Get<int32_t>(128);
```

#### 系统变量访问
 - GetBlockNum：获取当前任务配置的核数，用于代码内部的多核逻辑控制等。
 - GetBlockIdx：获取当前核的index，用于代码内部的多核逻辑控制及多核偏移量计算等。

## 高阶 API
### 数学函数
#### Tanh
 - 函数原型：__aicore__ inline void Tanh(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)，同样支持不使用sharedTmpBuffer的情况，接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。
 - 功能：计算srcTensor的tanh值，结果存储在dstTensor中。
 - 参数：
    - dstTensor：输出张量，存储tanh计算结果。
    - srcTensor：输入张量，存储待计算的元素。
    - sharedTmpBuffer：共享临时缓冲区，用于存储中间计算结果。
    - 可以添加calCount参数，指定需要计算的元素个数。
 - 支持类型：half, float

#### 其他数学函数
 - Asin
 - Sin
 - Cos
 - CumSum
 - Exp
 - 其他数学函数接口与Tanh类似，只是函数名不同。
 - 支持类型：half, float

### 激活函数
#### SoftMax
 - 函数原型1：接口框架申请临时空间，LocalTensor的数据类型相同 
```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```
 - 函数原型2：开发者自己申请临时空间，通过sharedTmpBuffer入参传入临时空间 
```cpp
template <typename T, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false, const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMax(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
```
 - 功能：计算srcTensor的softmax值，结果存储在dstTensor中。
 - 参数：
    - dstTensor：输出张量，存储softmax计算结果。
    - srcTensor：输入张量，存储待计算的元素。
    - sharedTmpBuffer：共享临时缓冲区，用于存储中间计算结果。
    - 其中sumTensor和maxTensor为临时张量，用于存储softmax计算过程中的中间结果。
 - 支持类型：half, float

#### Gelu
 - 函数原型：template <typename T, bool highPrecision = false, bool highPerformance = false> __aicore__ inline void Gelu(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const uint32_t dataSize)
 - 功能：计算srcTensor的gelu值，结果存储在dstTensor中。
 - 参数：
    - dstTensor：输出张量，存储gelu计算结果。
    - srcTensor：输入张量，存储待计算的元素。
    - dataSize：指定需要计算的元素个数。
 - 支持类型：half, float
 - 示例：
```cpp
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        AscendC::Gelu(dstLocal, srcLocal, dataSize);
        // AscendC::Gelu<srcType, true, false>(dstLocal, srcLocal, dataSize);
        // AscendC::Gelu<srcType, false, true>(dstLocal, srcLocal, dataSize);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
```

#### Swish
 - 函数原型：template <typename T, bool isReuseSource = false> __aicore__ inline void Swish(const LocalTensor<T> &dstLocal, const LocalTensor<T> &srcLocal, uint32_t dataSize, const T &scalarValue)
 - 功能：计算srcTensor的swish值，结果存储在dstTensor中。
 - 参数：
    - dstTensor：输出张量，存储swish计算结果。
    - srcTensor：输入张量，存储待计算的元素。
    - dataSize：指定需要计算的元素个数。
    - scalarValue：激活函数中的β参数。支持的数据类型为：half/float，β参数的数据类型需要与源操作数和目的操作数保持一致。
 - 支持类型：half, float
 - 示例：
 ```cpp
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
        AscendC::LocalTensor<srcType> srcLocal = inQueueX.DeQue<srcType>();
        AscendC::Swish(dstLocal, srcLocal, dataSize, scalarValue);
        outQueue.EnQue<srcType>(dstLocal);
        inQueueX.FreeTensor(srcLocal);
    }
 ```

#### Sigmoid
 - 函数原型：
  - 通过sharedTmpBuffer入参传入临时空间，也可以通过接口框架申请临时空间
    - 源操作数Tensor全部/部分参与计算
    template <typename T, bool isReuseSource = false>
    __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    - 源操作数Tensor全部参与计算template <typename T, bool isReuseSource = false> __aicore__ inline void Sigmoid(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
 - 功能：计算srcTensor的sigmoid值，结果存储在dstTensor中。
 - 参数：
    - dstTensor：输出张量，存储sigmoid计算结果。
    - srcTensor：输入张量，存储待计算的元素。
    - dataSize：指定需要计算的元素个数。
 - 支持类型：half, float
 - 示例：
 ```cpp
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQue;
    pipe.InitBuffer(tmpQue, 1, bufferSize);  // bufferSize 通过Host侧tiling参数获取
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer = tmpQue.AllocTensor<uint8_t>();
    // 输入shape信息为1024, 算子输入的数据类型为half, 实际计算个数为512
    AscendC::Sigmoid(dstLocal, srcLocal, sharedTmpBuffer, 512);
```

## 核函数开发 API
#### 核函数实现
```cpp
extern "C" __global__ __aicore__ void kernel_name(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR output, uint32_t totalLength)
{
    // 核函数实现
}
```

### 核函数调用
```cpp
    ACLRT_LAUNCH_KERNEL(kernel_name)(blockDim, acl_stream,
        cast<void*>(x.storage().data()), cast<void*>(y.storage().data()), cast<void*>(z.storage().data()), totalLength);
```
