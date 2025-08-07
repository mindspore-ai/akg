# ms_custom_ops - MindSpore 自定义算子框架

[![License](https://img.shields.io/badge/License-Apache%202.0lue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MindSpore](https://img.shields.io/badge/MindSpore-2.6-green.svg)](https://www.mindspore.cn/)

## 📖 概述

`ms_custom_ops` 是一个专为 MindSpore 设计的自定义算子开发框架，支持在昇腾 NPU 上高效实现自定义算子。该框架提供了完整的 PyNative 和 Graph 两种执行模式支持，内置缓存优化、内存管理等高级特性，让开发者能够快速构建高性能的自定义算子。

### ✨ 核心特性

- **双模式支持**: 同时支持 PyNative 动态执行和 Graph 静态编译模式
- **昇腾优化**: 专为昇腾 NPU 设计，充分利用硬件特性
- **缓存机制**: 内置算子缓存和 Tiling 缓存，显著提升性能
- **内存管理**: 自动管理设备内存和主机内存，确保内存安全
- **开发友好**: 提供完整的开发工具链和测试框架

## 系统架构

### 核心组件

1. **PyBoost 框架**：用于 PyNative 模式下的动态执行
2. **GraphMode 框架**：用于静态图编译模式
3. **共享组件**：包括内存管理、缓存优化等通用功能

### 核心模块说明

#### 1. ms_kernels_internal - 内部算子框架
- **pyboost/**: PyNative模式下的算子实现
  - `internal_pyboost_runner.h/cc`: PyBoost运行器基类，提供算子注册和执行框架
  - `internal_pyboost_utils.h/cc`: PyBoost工具函数，提供内存管理和缓存功能
  - `ops/`: 具体算子实现目录

- **graphmode/**: Graph模式下的算子实现
  - `internal_kernel_mod.h/cc`: 内部内核模块基类，提供Graph模式算子框架
  - `internal_kernel_utils.h/cc`: 内核工具函数
  - `internal_kernel_in_out_map.h/cc`: 输入输出映射管理
  - `ops/`: 具体算子实现目录

#### 2. ascendc - 昇腾C算子框架
- **kernel/**: 昇腾内核实现
  - `op_kernel/`: 设备端算子内核
  - `op_host/`: 主机端算子实现
- **pyboost/**: 昇腾PyBoost实现
- **graphmode/**: 昇腾Graph模式实现
- **op_compiler.py**: 算子编译器

#### 3. 共享基础设施
- **tiling_mem_mgr.h/cc**: Tiling内存管理器，负责设备内存分配和释放
- **internal_tiling_cache.h/cc**: 内部Tiling缓存，提供算子缓存和Tiling策略缓存
- **internal_helper.h/cc**: 内部辅助函数，提供通用工具函数
- **internal_spinlock.h**: 自旋锁实现，用于多线程同步

### 目录结构

```
ms_custom_ops/
├── src/                       # 源代码目录
│   ├── module.h               # 模块头文件
│   ├── module.cc              # 模块实现文件
│   ├── CMakeLists.txt         # CMake构建配置
│   ├── ms_kernels_internal/   # 内部算子实现
│   │   ├── CMakeLists.txt     # 内部算子构建配置
│   │   ├── internal_helper.h/cc        # 内部辅助函数
│   │   ├── internal_spinlock.h         # 自旋锁实现
│   │   ├── tiling_mem_mgr.h/cc         # Tiling内存管理器
│   │   ├── internal_tiling_cache.h/cc  # 内部Tiling缓存
│   │   ├── pyboost/           # PyNative模式实现
│   │   │   ├── internal_pyboost_runner.h/cc    # PyBoost运行器基类
│   │   │   ├── internal_pyboost_utils.h/cc     # PyBoost工具函数
│   │   │   └── ops/           # PyBoost算子实现
│   │   │       └── reshape_and_cache_runner.cc  # reshape_and_cache算子
│   │   └── graphmode/         # Graph模式实现
│   │       ├── internal_kernel_mod.h/cc         # 内部内核模块基类
│   │       ├── internal_kernel_utils.h/cc       # 内部内核工具函数
│   │       ├── internal_kernel_in_out_map.h/cc  # 输入输出映射
│   │       └── ops/           # Graph模式算子实现
│   │           └── reshape_and_cache.cc         # reshape_and_cache算子
│   ├── ascendc/               # 昇腾C相关组件
│   │   ├── CMakeLists.txt     # 昇腾C构建配置
│   │   ├── op_compiler.py     # 算子编译器
│   │   ├── kernel/            # 昇腾内核实现
│   │   │   ├── op_kernel/     # 算子内核
│   │   │   │   └── add_custom.cpp      # 自定义加法算子
│   │   │   └── op_host/       # 算子主机端
│   │   │       ├── add_custom.cpp      # 主机端加法算子
│   │   │       └── add_custom_tiling.h # 加法算子Tiling配置
│   │   ├── pyboost/           # 昇腾PyBoost实现
│   │   │   ├── ascendc_pyboost_runner.h # 昇腾PyBoost运行器
│   │   │   └── ops/           # 昇腾PyBoost算子
│   │   │       └── add_runner.cc       # 加法算子运行器
│   │   └── graphmode/         # 昇腾Graph模式实现
│   │       ├── ascendc_kernel_mod.h/cc # 昇腾内核模块
│   │       └── ops/           # 昇腾Graph模式算子
│   │           └── add.cc             # 加法算子实现
│   └── swft/                  # SWFT相关组件（预留）
├── yaml/                      # 算子描述yaml目录
│   ├── ascendc/               # 昇腾算子yaml
│   │   └── add_op.yaml        # 加法算子配置
│   └── ms_kernels_internal/   # 内部算子yaml
│       └── reshape_and_cache_op.yaml  # reshape_and_cache算子配置
├── python/                    # Python包目录
│   └── ms_custom_ops/         # 主包目录
│       └── __init__.py        # 包初始化文件
├── tests/                     # 测试目录
│   ├── test_add.py            # 加法算子测试
│   └── test_custom_reshape_and_cache.py  # reshape_and_cache算子测试
├── build/                     # 构建输出目录
├── dist/                      # 分发目录
├── setup.py                   # 安装脚本
├── requirements.txt           # Python依赖
├── version.txt                # 版本信息
├── .gitignore                 # Git忽略文件
├── .commit_id                 # 提交ID文件
└── README.md                  # 项目说明文档
```

## 快速开始

### 1. 环境准备

确保已安装：
- MindSpore >= 2.6
- 昇腾 CANN 工具包
- CMake >= 3.14
- Python >= 3.9

### 2. 安装编译

```bash
# 克隆代码仓库
git clone <repository_url>
cd ms_custom_ops

# 安装（会自动编译自定义算子）
python setup.py install
```

编译过程会自动：
- 检测昇腾环境
- 使用 CMake 构建自定义算子
- 将生成的 .so 文件安装到正确位置

### 3. 使用示例

#### PyNative 模式

```python
import mindspore as ms
import ms_custom_ops

# 设置为 PyNative 模式
ms.set_context(mode=ms.context.PYNATIVE_MODE)
ms.set_device("Ascend")

# 直接调用自定义算子
output = ms_custom_ops.reshape_and_cache(
    key, value, key_cache, value_cache, slot_mapping, head_num
)
```

#### Graph 模式

```python
import mindspore as ms
from mindspore.ops import ModuleWrapper
import ms_custom_ops

# 设置为 Graph 模式
ms.set_context(mode=ms.context.GRAPH_MODE)
ms.set_device("Ascend")

# 使用 ModuleWrapper 封装
class MyNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        
    def construct(self, key, value, key_cache, value_cache, slot_mapping, head_num):
        return ms_custom_ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, head_num)

# 使用网络
net = MyNet()
output = net(key, value, key_cache, value_cache, slot_mapping, head_num)
```

## 开发自定义算子

### 开发流程概览

开发一个新的自定义算子需要以下步骤：

1. **设计算子接口** - 确定输入输出和参数
2. **实现算子逻辑** - 编写PyBoost和GraphMode实现
3. **添加配置文件** - 创建YAML配置文件
4. **编写测试用例** - 创建单元测试
5. **编译和验证** - 构建并测试算子

### 1. 创建算子实现



#### PyBoost 模式实现

在 `ms_custom_ops/src/ms_kernels_internal/pyboost/ops/` 下创建新文件：

```cpp
// my_op_runner.cc
#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {
class MyOpRunner : public InternalPyboostRunner {
public:
    using InternalPyboostRunner::InternalPyboostRunner;

protected:
    internal::InternalOpPtr
    CreateKernel(const internal::InputsImmutableInfoList &inputs,
                 const internal::OutputsImmutableInfoList &outputs) override {
        // 创建内部算子，这里需要根据具体算子实现
        // 例如：return internal::CreateMyOp(inputs, outputs, param, internal::kInternalMyOpName);
        return nullptr;
    }

    void LaunchKernel() {
        tensor::TensorPtrList inputs;
        inputs.reserve(2); // 根据实际输入数量调整

        for (const auto &input : this->inputs()) {
            inputs.push_back(input.is_defined() ? input.tensor() : nullptr);
        }

        tensor::TensorPtrList outputs;
        TransInternalShapes(inputs, outputs);
        LAUNCH_INTERNAL(_op_name_, this->_device_context_, this->stream_id(),
                        inputs, outputs);
    }
};

// 注册算子名称映射
MS_KERNELS_INTERNAL_FACTORY_REG(MyOp, internal::kInternalMyOpName);
} // namespace ms::pynative

namespace ms_custom_ops {
// 辅助函数：生成结果张量
ms::Tensor GenResultTensor(const ms::Tensor &input) {
    return ms::Tensor(input.data_type(), input.shape());
}

// 主要算子函数
ms::Tensor npu_my_op(const ms::Tensor &input1, const ms::Tensor &input2) {
    auto result = GenResultTensor(input1);
    auto op_name = "MyOp";
    auto runner = std::make_shared<ms::pynative::MyOpRunner>(op_name);

    // 设置参数（如果需要）
    // runner->SetParam(param_value);

    // 转换为 TensorPtr 用于 hash 计算
    auto input1_tensor_ptr = input1.tensor();
    auto input2_tensor_ptr = input2.tensor();

    // 设置运行器参数（包括 hash 计算）
    runner->Setup(op_name, input1_tensor_ptr, input2_tensor_ptr);

    // 运行操作
    runner->Run({input1, input2}, {result});
    return result;
}
} // namespace ms_custom_ops

// PyBoost 调用函数
auto pyboost_my_op(const ms::Tensor &input1, const ms::Tensor &input2) {
    return ms::pynative::PyboostRunner::Call<1>(
        ms_custom_ops::npu_my_op, input1, input2);
}

// 注册到 Python 模块
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
    m.def("my_op", &pyboost_my_op, "My Custom Operator",
          pybind11::arg("input1"), pybind11::arg("input2"));
}
```

**重要说明**：
- PyBoost 算子需要继承 `InternalPyboostRunner` 并实现 `CreateKernel` 方法
- 需要实现 `LaunchKernel` 方法来处理具体的执行逻辑
- 使用 `MS_KERNELS_INTERNAL_FACTORY_REG` 注册算子名称映射
- 需要提供 `npu_my_op` 函数作为主要算子实现
- 使用 `pyboost_my_op` 函数作为 PyBoost 调用接口
- 使用 `MS_CUSTOM_OPS_EXTENSION_MODULE` 注册到 Python 模块

#### GraphMode 实现

在 `ms_custom_ops/src/ms_kernels_internal/graphmode/ops/` 下创建新文件：

```cpp
// my_op.cc
#include "ms_custom_ops/src/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/op_func_impl.h"

namespace mindspore {
namespace ops {
class OPS_API CustomMyOpFuncImpl : public OpFuncImpl {
public:
    ShapeArray InferShape(const PrimitivePtr &primitive,
                          const InferInfoPtrList &input_infos) const override {
        return {input_infos[0]->GetShape()};
    }
    std::vector<TypeId>
    InferType(const PrimitivePtr &primitive,
              const InferInfoPtrList &input_infos) const override {
        return {input_infos[0]->GetType()};
    }

    bool GeneralInferRegistered() const override { return true; }
};
} // namespace ops
} // namespace mindspore

namespace ms_custom_ops {
class CustomMyOp : public InternalKernelMod {
public:
    CustomMyOp() : InternalKernelMod() {}
    ~CustomMyOp() = default;

protected:
    internal::InternalOpPtr
    CreateKernel(const internal::InputsImmutableInfoList &inputs,
                 const internal::OutputsImmutableInfoList &outputs,
                 const std::vector<KernelTensor *> &ms_inputs,
                 const std::vector<KernelTensor *> &ms_outputs) override {
        // 创建内部算子，这里需要根据具体算子实现
        // 例如：return internal::CreateMyOp(inputs, outputs, param, internal::kInternalMyOpName);
        return nullptr;
    }
};

// 注册算子名称映射
MS_CUSTOM_INTERNAL_KERNEL_NAME_REG(my_op, internal::kInternalMyOpName);

// 注册输入输出索引映射（根据实际输入数量调整）
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(my_op, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(my_op, OUTPUT_NUM_1, INDEX_0);

} // namespace ms_custom_ops

// 注册算子到 MindSpore 框架
MS_CUSTOM_OPS_REGISTER(my_op, CustomMyOpFuncImpl, CustomMyOp);
```

**重要说明**：
- GraphMode 算子需要实现 `CreateKernel` 方法来创建内部算子
- 基类 `InternalKernelMod` 已经实现了 `Resize` 和 `Launch` 的通用逻辑
- 需要正确注册算子名称映射和输入输出索引映射
- 如果算子需要额外的工作空间，可以在 `UpdateParam` 中设置 `workspace_size_list_`
- 算子需要同时实现 `OpFuncImpl` 类来处理形状和类型推断

### 2. 添加 Python 接口

在 `ms_custom_ops/__init__.py` 中添加：

```python
def my_op(*args, **kwargs):
    """My custom operator"""
    return ops.Custom(func_type="internal", func_name="MyOp", out_shape=..., out_dtype=...)(*args, **kwargs)
```

### 3. 添加配置文件

在 `yaml/ms_kernels_internal/` 下创建算子配置文件：

```yaml
# my_op.yaml
op_name: "MyOp"
func_name: "MyOp"
input_names: ["input1", "input2"]
output_names: ["output"]
input_dtypes: ["float16", "float16"]
output_dtypes: ["float16"]
input_shapes: ["dynamic", "dynamic"]
output_shapes: ["dynamic"]
```

### 4. 编写测试

创建测试文件 `tests/test_my_op.py`：

```python
import pytest
import numpy as np
import mindspore as ms
import ms_custom_ops

@pytest.mark.parametrize('exec_mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_my_op(exec_mode):
    ms.set_context(mode=exec_mode)
    ms.set_device("Ascend")
    
    # 准备输入数据
    input_data = np.random.rand(10, 20).astype(np.float16)
    
    # 执行算子
    output = ms_custom_ops.my_op(ms.Tensor(input_data))
    
    # 验证结果
    expected = # 计算期望结果
    assert np.allclose(output.asnumpy(), expected, rtol=1e-3, atol=1e-3)
```

## 高级特性

### 1. 双模式执行机制

#### GraphMode Resize 接口机制

GraphMode 算子中的 `Resize` 接口是处理动态形状变化的核心机制：

#### 基类 Resize 功能
`InternalKernelMod` 基类的 `Resize` 方法自动处理：
- **形状更新**：将输入输出张量的形状信息转换为内部格式
- **内核重建**：当参数变化时自动重建内部算子内核
- **Tiling 缓存**：智能缓存和复用 Tiling 策略
- **内存管理**：自动管理工作空间内存分配

#### 自定义 Resize 逻辑
子类通常不需要重写 `Resize` 方法，基类已经处理了所有通用逻辑。如果需要添加特定逻辑，可以重写 `UpdateParam` 方法：

```cpp
bool UpdateParam(const std::vector<KernelTensor*> &inputs,
                 const std::vector<KernelTensor*> &outputs) override {
    // 验证输入形状
    auto input_shape = inputs[0]->GetShapeVector();
    if (input_shape.size() != 3) {
        MS_LOG(ERROR) << "Input shape must be 3D";
        return false;
    }
    
    // 设置工作空间大小（如果需要）
    workspace_size_list_ = {input_shape[0] * input_shape[1] * sizeof(float)};
    
    return true;
}
```

#### PyBoost 动态执行机制

PyBoost 模式下的算子执行采用动态方式：

```cpp
// 主要执行流程
void LaunchKernel() {
    // 1. 准备输入输出张量
    tensor::TensorPtrList inputs;
    for (const auto &input : this->inputs()) {
        inputs.push_back(input.is_defined() ? input.tensor() : nullptr);
    }
    
    // 2. 转换形状信息
    tensor::TensorPtrList outputs;
    TransInternalShapes(inputs, outputs);
    
    // 3. 启动内核执行
    LAUNCH_INTERNAL(_op_name_, this->_device_context_, this->stream_id(),
                    inputs, outputs);
}
```

**PyBoost 特点**：
- **动态执行**：每次调用都会重新计算 hash 和创建内核
- **自动缓存**：框架自动缓存相同配置的算子实例
- **内存管理**：自动管理工作空间内存的分配和释放
- **异步执行**：支持异步执行和流管理

### 2. Hash 缓存优化

框架自动为算子提供基于 hash 的缓存机制：

- **算子缓存**：避免重复创建相同配置的算子
- **Tiling 缓存**：缓存切分策略，加速执行

### 2. 内存管理

- 自动管理输入、输出和工作空间内存
- 支持设备内存和主机内存
- 引用计数机制确保内存安全

### 3. 性能优化建议

1. **使用缓存**：充分利用框架提供的缓存机制
2. **批量处理**：设计算子时考虑批量数据处理
3. **内存复用**：合理规划工作空间大小

## 调试技巧

### 1. 日志输出

设置环境变量开启详细日志：
```bash
export GLOG_v=3
```

### 2. 性能分析

使用 MindSpore Profiler 分析算子性能：
```python
from mindspore.profiler import Profiler

profiler = Profiler()
# 执行算子
profiler.analyse()
```

### 3. 常见问题

**Q: Resize 接口返回 KRET_RESIZE_FAILED**  
A: 检查以下几点：
1. 确保 `CreateKernel` 方法正确实现并返回有效的内部算子
2. 验证 `UpdateParam` 方法是否正确处理参数
3. 检查输入输出索引映射是否正确注册
4. 查看日志确认具体的失败原因

**Q: 编译失败提示找不到 CANN 环境**  
A: 确保正确安装昇腾 CANN 工具包，并设置环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Q: 算子在不同模式下行为不一致**  
A: 检查是否正确处理了 Parameter 和 Tensor 的区别，Graph 模式下缓存通常使用 Parameter。

**Q: 性能不如预期**  
A: 1) 检查是否正确使用了缓存机制；2) 确认内存访问模式是否高效；3) 使用 Profiler 定位瓶颈。

**Q: PyBoost 模式下算子执行失败**  
A: 检查以下几点：
1. 确保 `CreateKernel` 方法正确实现并返回有效的内部算子
2. 验证 `LaunchKernel` 方法中的张量处理逻辑
3. 检查 `Setup` 方法中的参数设置和 hash 计算
4. 确认 Python 模块注册是否正确

## 示例：reshape_and_cache 算子

reshape_and_cache 是一个典型的自定义算子示例，用于 KV Cache 的更新操作：

### 功能描述
- 将输入的 key 和 value 张量 reshape 后写入到指定的缓存位置
- 支持灵活的 slot 映射机制
- 高效的内存更新操作

### 使用方法
```python
# 参数说明
# key: 输入的 key 张量，shape 为 (batch, seq_len, hidden_dim) 或 (batch*seq_len, hidden_dim)
# value: 输入的 value 张量，shape 同 key
# key_cache: key 缓存张量，shape 为 (num_slots, slot_size, num_heads, head_dim)
# value_cache: value 缓存张量，shape 同 key_cache
# slot_mapping: 指定每个 token 写入的 slot 位置
# head_num: attention head 数量

output = ms_custom_ops.reshape_and_cache(
    key, value, key_cache, value_cache, slot_mapping, head_num
)
```

## 文件命名规范

为了保持项目结构的一致性，请遵循以下命名规范：

### 算子实现文件
- **PyBoost模式**: `{op_name}_runner.cc` (如: `reshape_and_cache_runner.cc`)
- **GraphMode模式**: `{op_name}.cc` (如: `reshape_and_cache.cc`)
- **昇腾算子**: `{op_name}_custom.cpp` (如: `add_custom.cpp`)

### 配置文件
- **YAML配置**: `{op_name}_op.yaml` (如: `reshape_and_cache_op.yaml`)

### 测试文件
- **测试文件**: `test_{op_name}.py` (如: `test_reshape_and_cache.py`)

### 头文件
- **基类头文件**: 使用描述性名称 (如: `internal_pyboost_runner.h`)
- **工具头文件**: 使用功能描述 (如: `internal_helper.h`)

## 贡献指南

欢迎贡献新的自定义算子！请遵循以下步骤：

1. Fork 代码仓库
2. 创建特性分支
3. 实现算子并添加测试
4. 提交 Pull Request

确保：
- 代码符合项目编码规范
- 添加充分的单元测试
- 更新相关文档
- 遵循文件命名规范

## 许可证

本项目采用 Apache License 2.0 许可证。