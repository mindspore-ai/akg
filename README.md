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
├── ccsrc/                        # C++核心源码
│   ├── base/                     # 基础设施
│   │   ├── ms_kernels_internal/  # 内部算子基础
│   │   │   ├── pyboost/          # PyNative模式基类/工具
│   │   │   ├── graphmode/        # Graph模式基类/工具
│   │   │   ├── tiling_mem_mgr.h/cc
│   │   │   ├── internal_helper.h/cc
│   │   │   ├── internal_spinlock.h
│   │   │   ├── internal_tiling_cache.h/cc
│   │   └── ascendc/              # 昇腾算子基础
│   │       ├── pyboost/
│   │       ├── graphmode/
│   ├── ops/                      # 算子实现
│   │   ├── ms_kernels_internal/
│   │   │   └── reshape_and_cache.cc
│   │   ├── ascendc/
│   │   │   ├── add.cc
│   │   │   ├── kernel_impl/
│   │   │   │   ├── op_kernel/
│   │   │   │   │   └── add_custom.cpp
│   │   │   │   ├── op_host/
│   │   │   │   │   ├── add_custom.cpp
│   │   │   │   │   └── add_custom_tiling.h
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── module.h
│   └── module.cc
├── python/
│   └── ms_custom_ops/
│       └── __init__.py
├── yaml/
│   ├── ascendc/
│   │   └── add_op.yaml
│   └── ms_kernels_internal/
│       └── reshape_and_cache_op.yaml
├── tests/
│   ├── test_add.py
│   └── test_custom_reshape_and_cache.py
├── build/
├── dist/
├── setup.py
├── requirements.txt
├── version.txt
├── .gitignore
├── .commit_id
└── README.md
```

## 快速开始

### 1. 环境准备

确保已安装：
- MindSpore br_infer_iter分支日构建包
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
};

// 注册算子名称映射
MS_KERNELS_INTERNAL_NAME_REG(MyOp, internal::kInternalMyOpName);
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

    // 设置参数（如果需要将部分输入转为属性）
    // runner->SetParam(param_value);

    // 设置运行器参数（包括 hash 计算）
    runner->Setup(op_name, input1, input2);

    runner->GetOrCreateKernel(inputs, outputs);

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

    // 输入和输出和底层算子的映射
    void InitKernelInputsOutputsIndex() override {
        kernel_inputs_index_ = {kInputKeyIndex, kInputValueIndex, kInputKeyCacheIndex,
                                kInputValueCacheIndex, kInputSlotMappingIndex};
        kernel_outputs_index_ = {kOutputIndex};
    }

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
} // namespace ms_custom_ops

// 注册算子到 MindSpore 框架
// 注册算子名称映射 (对外接口my_op, 内部算子库名字internal::kInternalMyOpName
   对接的kernelmod CustomMyOp)
REG_GRAPH_MODE_OP(my_op, internal::kInternalMyOpName,
                       CustomMyOp);
```

**重要说明**：
- GraphMode 算子需要实现 `CreateKernel` 方法来创建内部算子
- 基类 `InternalKernelMod` 已经实现了 `Resize` 和 `Launch` 的通用逻辑
- 需要正确注册算子名称映射和输入输出索引映射
- 算子需要同时实现 `OpFuncImpl` 类来处理形状和类型推断

### 2. 添加配置文件

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

### 3. 编写测试

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