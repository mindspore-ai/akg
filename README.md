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

## 🏗️ 项目架构

### 核心组件

1. **PyBoost 框架**：用于 PyNative 模式下的动态执行
2. **GraphMode 框架**：用于静态图编译模式
3. **共享组件**：包括内存管理、缓存优化等通用功能

### 核心模块说明

#### 1. ms_kernels_internal - 内部算子框架
- **ccsrc/base/ms_kernels_internal/pyboost**: PyNative模式下的算子公共基类实现
- **ccsrc/base/ms_kernels_internal/graphmode**: Graph模式下的算子公共基类实现
- **ccsrc/ops/ms_kernels_internal/*.cc**: 算子调用实现

- **公共文件**:
  - **tiling_mem_mgr.h/cc**: Tiling内存管理器，负责设备内存分配和释放
  - **internal_tiling_cache.h/cc**: 内部Tiling缓存，提供算子缓存和Tiling策略缓存
  - **internal_helper.h/cc**: 内部辅助函数，提供通用工具函数
  - **internal_spinlock.h**: 自旋锁实现，用于多线程同步

#### 2. ascendc - 昇腾C算子框架
- **ccsrc/base/ascendc/pyboost**: PyNative模式下的算子公共基类实现
- **ccsrc/base/ascendc/graphmode**: Graph模式下的算子公共基类实现
- **ccsrc/ops/ascendc/**: 算子kernel和调用实现

### 目录结构

```
ms_custom_ops/
├── ccsrc/                        # C++核心源码
│   ├── base/                     # 基础设施
│   │   ├── ms_kernels_internal/  # 内部算子基础框架
│   │   │   ├── pyboost/          # PyNative模式基类/工具
│   │   │   ├── graphmode/        # Graph模式基类/工具
│   │   │   ├── tiling_mem_mgr.h/cc      # Tiling内存管理
│   │   │   ├── internal_helper.h/cc      # 内部辅助函数
│   │   │   ├── internal_spinlock.h       # 自旋锁实现
│   │   │   └── internal_tiling_cache.h/cc # 内部Tiling缓存
│   │   └── ascendc/              # 昇腾算子基础
│   │       ├── pyboost/
│   │       └── graphmode/
│   ├── ops/                      # 算子实现
│   │   ├── ms_kernels_internal/
│   │   │   └── {op_name}.cc
│   │   ├── ascendc/
│   │   │   ├── {op_name}/
│   │   │   │   ├── {op_name}.cc
│   │   │   │   ├── op_host/
│   │   │   │   └── op_kernel/
│   │   │   └── CMakeLists.txt
│   │   └── CMakeLists.txt
│   ├── CMakeLists.txt
│   ├── module.h
│   └── module.cc
├── cmake/                        # CMake配置文件
│   ├── compile_ascendc_ops.cmake
│   └── find_ms_internal_kernels_lib.cmake
├── python/
│   └── ms_custom_ops/
│       └── __init__.py
├── yaml/                         # 算子配置文件
│   ├── ascendc/
│   |   └── {op_name}_op.yaml
│   ├── doc/
│   |   ├── {op_name}.md          # Markdown源文件
│   |   └── {op_name}_doc.yaml    # 生成的文档YAML文件
│   └── ms_kernels_internal/
│   |   └── {op_name}_op.yaml
├── tests/                        # 测试文件
│   └── st/
├── scripts/
│   └── op_compiler.py
├── build.sh                      # 一键编译脚本
├── setup.py                      # Python安装配置
├── requirements.txt               # Python依赖
├── version.txt                   # 版本信息
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

确保已安装：
- **MindSpore**: br_infer_iter分支日构建包
- **昇腾 CANN 工具包**: 最新版本
- **CMake**: >= 3.16
- **Python**: >= 3.9
- **Git**: 用于获取提交信息

### 2. 环境配置

```bash
# 设置昇腾环境变量
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
```

### 3. 编译安装

#### 方法一：使用 build.sh 脚本（推荐）

```bash
# 查看编译选项
bash build.sh -h

# 默认编译（Release模式）
bash build.sh

# Debug编译
bash build.sh -d

# 编译指定算子
bash build.sh -p ${absolute_op_dir_path}

# 编译指定算子
bash build.sh -p ${absolute_op_dir_path}
eg. bash build.sh -p /home/ms_custom_ops/ccsrc/ops/ascendc/add,/home/ms_custom_ops/ccsrc/ops/ascendc/add_rms_norm

# 指定SOC Verison编译
eg. bash build.sh -v ascend910b4
```

#### 方法二：使用 setup.py 安装

```bash
# 安装（会自动编译自定义算子）
python setup.py install

# 或者构建wheel包
python setup.py bdist_wheel
```

编译过程会自动：
- 检测昇腾环境
- 使用 CMake 构建自定义算子
- 将生成的 .so 文件安装到正确位置

## 📝 使用示例

### PyNative 模式

```python
import mindspore as ms
import ms_custom_ops

# 直接调用自定义算子
output = ms_custom_ops.reshape_and_cache(
    key, value, key_cache, value_cache, slot_mapping, head_num
)
```

### Graph 模式

```python
import mindspore as ms
import ms_custom_ops

reshape_and_cache = ms.jit(func=ms_custom_ops.reshape_and_cache)
output = reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, head_num)
```

## 开发自定义算子

### 开发流程概览

开发一个新的自定义算子需要以下步骤：

1. **设计算子接口** - 确定输入输出和参数，编写算子前端接口定义
2. **实现算子逻辑** - 编写PyBoost和GraphMode实现
3. **编写测试用例** - 创建单元测试
4. **编译和验证** - 构建并测试算子

#### 1. 设计算子接口

需要明确算子输入输出类型，并根据确定的算子接口，编写算子前端定义的YAML文件，并实现{op_name}_op.yaml和{op_name}_doc.yaml。
{op_name}_doc.yaml是算子参数说明文件，{op_name}_op.yaml是算子接口定义文件，其内容格式介绍：
```yaml
# Defining the function name and Primitive name of operators, use the '_' to separate words. For example, op_name is 'word1_word2', then the function name is 'word1_word2', and the Primitive class name is 'Word1Word2'.
<op_name>:
  # The 'args' is a fixed key of yaml file to define input args of operators.
  <args>:
    # Mandatory. For every arg, key is operators' argument name, and the value are some items, items' key name can be 'dtype', 'prim_init', 'default', 'type_cast','arg_handler'.
    <arg1>:
      # Mandatory. The 'dtype' is a fixed key.
      # Value is one of {int, float, bool, number, tensor, tuple, list, tuple[int], tuple[float], tuple[bool], tuple[number], tuple[tensor], list[int], list[float], list[bool], list[number], list[tensor]}.
      # If value is 'number', arg can be 'int', 'float' or 'bool'.
      <dtype>: <value>

      # Optional. The 'default' is a fixed key.
      # This item means input arg can use default value.
      # If arg_handler is not empty, the type of default value should be the first one of 'arg_handler_map'
      <default>: <value>

      # Optional. The 'prim_init' is a fixed key. Value can be 'True' or 'False', arg is arg of '__init__' of Primitive if value is 'True'.
      <prim_init>: <value>

      # Optional. The 'type_cast' is a fixed key. This item means can accept unmatchable input by implicit conversion. Value is one of {int, float, bool, number, tensor, tuple, list, tuple[int], tuple[float], tuple[bool], tuple[number], tuple[tensor], list[int], list[float], list[bool], list[number], list[tensor]}
      # Supported type cast now:
      # 1. int, float, bool, number <-> tensor.
      # 2. int, float, bool, number, tensor <-> list/tuple.
      # 3. list <-> tuple.
      <type_cast>: <value>

      # Optional. The 'arg_handler' is a fixed key. Value is a function name used to convert arg. For example, converting kernel size from 2 to (2, 2).
      <arg_handler>: <value>

    <arg2>:
      ...

    <args_signature>: #Optional
      # Optional. The 'rw_write' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_write>: <arg_name>

      # Optional. The 'rw_read' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_read>: <arg_name>

      # Optional. The 'rw_ref' is a fixed key, 'arg_name' is the corresponding arg name.
      <rw_ref>: <arg_name>

      # Optional. arg1 and arg2 should has same dtype. arg3 and arg4 should has same dtype.
      <dtype_group>: (<arg_name1>, <arg_name2>, ...), (<arg_name3>, <arg_name4>, ...), ...

    # The 'returns' is a fixed key of yaml file to define output of operators.
    <returns>:
      # Mandatory. For every output, key is operators' output name, and the value is a item, item's key is 'dtype'.
      <output1>:
        # Mandatory. Just refer to key 'dtype' in args.
        <dtype>: <value>

        # Optional. The 'inplace' is a fixed key. Value is input name of operator if the input is a inplace input.
        <inplace>: <value>

      <output2>:
        ...

    # Optional. The 'view' is a fixed key. Value should be set as 'True' if this is a view operator.
    # Default: False.
    <view>: <value>
```
具体可参考`yaml/ms_kernels_internal/reshape_and_cache_op.yaml`和`yaml/doc/reshape_and_cache_doc.yaml`

#### 2. 实现算子逻辑

#### ascendc算子

创建自定义算子目录：`ccsrc/ops/ascendc/{op_name}`，其中`ccsrc/ops/ascendc/{op_name}/op_kernel`和`ccsrc/ops/ascendc/{op_name}/op_host`是ascendc算子kernel实现，算子会被编译成aclnn的两段式接口。  
在`ccsrc/ops/ascendc/{op_name}/{op_name}.cc`文件实现算子kernel的调用。算子的pyboost和graph调用实现均在此文件。
要实现的主要类和函数：  
graph:  
1）算子infer函数，用来推导算子输出shape和dtype；  
2）算子KernelMod，需要继承`AscendCKernelMod`并重写`Launch`和`GetWorkSpaceInfo`接口；  

pyboost:  
1）算子kernel调用函数;   
2）pybind接口注册；  

以add算子为例：
```cpp
#include "ascendc_kernel_mod.h"
#include "ms_extension/api.h"
#include <map>
#include <string>
#include <vector>

// =============================================================================
// 图模式调用实现
// =============================================================================

namespace ms_custom_ops {
// 算子infer函数，需要实现InferShape和InferType函数
class OPS_API AddCustomOpFuncImpl : public OpFuncImpl {
public:
  // 算子infershape，需要返回算子所有输出的shape大小
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    auto out_shape = input_infos[0]->GetShape();
    return {out_shape};
  }

  // 算子infertype，需要返回算子所有输出的数据类型
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

// 算子graph模式调用，aclnn两段式接口调用，需要实现Launch和GetWorkSpaceInfo函数
class AddCustomAscend : public AscendCKernelMod {
public:
  AddCustomAscend() : AscendCKernelMod(std::move("aclnnAddCustom")) {}
  ~AddCustomAscend() = default;

  // 算子执行调用函数，RunOp函数会调用aclnn算子第二段接口
  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override {
    MS_EXCEPTION_IF_NULL(stream_ptr);
    RunOp(stream_ptr, workspace, inputs[0], inputs[1], outputs[0]);
    return true;
  }

  // 算子workspace调用函数，GetWorkspaceForResize会调用aclnn算子第一段接口
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override {
    GetWorkspaceForResize(inputs[0], inputs[1], outputs[0]);
  }

private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE();
};
} // namespace ms_custom_ops

// 注册算子infer函数
REG_GRAPH_MODE_OP(add, ms_custom_ops::AddCustomOpFuncImpl,
                  ms_custom_ops::AddCustomAscend);

// =============================================================================
// PYBOOST调用实现
// =============================================================================

#include "ascendc_pyboost_runner.h"

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::device::ascend;
// 算子kernel调用函数，需要手动创建输出tensor
ms::Tensor custom_add(const ms::Tensor &x, const ms::Tensor &y) {
  // 创建输出空tensor
  auto out = ms::Tensor(x.data_type(), x.shape());
  // 初始化runner运行器
  auto runner = std::make_shared<ms::pynative::AscendCOpRunner>("AddCustom");
  // 设置runner需要具体执行的函数，由LAUNCH_ASCENDC_FUNC封装了aclnn接口调用
  runner->SetLaunchFunc(LAUNCH_ASCENDC_FUNC(aclnnAddCustom, x, y, out));
  // 执行runner
  runner->Run({x, y}, {out});
  return out;
}

// pybind调用函数
auto pyboost_add(const ms::Tensor &x, const ms::Tensor &y) {
  // Call<输出个数>
  return ms::pynative::PyboostRunner::Call<1>(custom_add, x, y);
}
} // namespace ms_custom_ops

// 算子接口注册，对接C++和python接口
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("add", &ms_custom_ops::pyboost_add, "add", pybind11::arg("x"),
        pybind11::arg("y"));
}
```

#### internal算子

在`ccsrc/ops/ms_kernels_internal/{op_name}.cc`文件实现算子kernel的调用。算子的pyboost和graph调用实现均在此文件。
要实现的主要类和函数：  
graph：  
1）算子infer函数，用来推导算子输出shape和dtype；  
2）算子KernelMod，需要继承`InternalKernelMod`并重写`InitKernelInputsOutputsIndex`和`CreateKernel`接口； 

pyboost:  
1）算子kernel调用函数;  
2）pybind接口注册；  

以reshape_and_cache算子为例：
```cpp
#include "ccsrc/base/ms_kernels_internal/graphmode/internal_kernel_mod.h"
#include "ms_extension/api.h"
#include "ccsrc/utils/utils.h"

namespace ms_custom_ops {

// =============================================================================
// 图模式调用实现
// =============================================================================

// 1. 算子infer函数
class OPS_API CustomReshapeAndCacheOpFuncImpl : public OpFuncImpl {
public:
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetShape()}; // 输出shape与第一个输入相同
  }

  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()}; // 输出类型与第一个输入相同
  }
  
  bool GeneralInferRegistered() const override { return true; }
};

// 2. 算子KernelMod
class CustomReshapeAndCache : public InternalKernelMod {
public:
  CustomReshapeAndCache() : InternalKernelMod(), skip_execution_(false) {}
  ~CustomReshapeAndCache() = default;

  void InitKernelInputsOutputsIndex() override {
    // 指定参与计算的输入输出索引
    kernel_inputs_index_ = {0, 1, 2, 3, 4}; // key, value, key_cache, value_cache, slot_mapping
    kernel_outputs_index_ = {0};
  }

  // 重写Resize处理零维度输入
  int Resize(const std::vector<KernelTensor *> &inputs, 
             const std::vector<KernelTensor *> &outputs) override {
    // 检查输入是否包含0维度，如果有则跳过执行
    for (const auto &input : inputs) {
      if (input == nullptr) continue;
      auto shape = input->GetShapeVector();
      for (const auto &dim : shape) {
        if (dim == 0) {
          skip_execution_ = true;
          return KernelMod::Resize(inputs, outputs);
        }
      }
    }
    skip_execution_ = false;
    return InternalKernelMod::Resize(inputs, outputs);
  }

  // 重写Launch处理跳过执行标志
  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, 
              void *stream_ptr) override {
    if (skip_execution_) {
      return true; // 跳过执行，直接返回成功
    }
    return InternalKernelMod::Launch(inputs, workspace, outputs, stream_ptr);
  }

protected:
  internal::InternalOpPtr CreateKernel(
      const internal::InputsImmutableInfoList &inputs,
      const internal::OutputsImmutableInfoList &outputs,
      const std::vector<KernelTensor *> &ms_inputs,
      const std::vector<KernelTensor *> &ms_outputs) override {
    // 从输入张量中提取参数
    internal::ReshapeAndCacheParam param;
    auto head_num = ms_inputs.at(6); // head_num在第6个位置
    param.head_num = static_cast<int32_t>(head_num->GetValue<int64_t>().value());
    
    auto cache_mode = ms_inputs.at(5); // cache_mode在第5个位置
    int32_t cache_mode_val = static_cast<int32_t>(cache_mode->GetValue<int64_t>().value());

    // 根据cache_mode设置格式：NZ格式需要特殊处理
    if (cache_mode_val == 1) { // NZ格式
      auto inputs_clone = inputs;
      inputs_clone[2].SetFormat(internal::kFormatFRACTAL_NZ); // key_cache
      inputs_clone[3].SetFormat(internal::kFormatFRACTAL_NZ); // value_cache
      return internal::CreateAsdReshapeAndCacheOp(inputs_clone, outputs, param,
                                                  internal::kInternalAsdReshapeAndCacheOpName);
    }
    return internal::CreateAsdReshapeAndCacheOp(inputs, outputs, param, 
                                                internal::kInternalAsdReshapeAndCacheOpName);
  }

private:
  bool skip_execution_; // 跳过执行标志
};
} // namespace ms_custom_ops

// 注册算子
REG_GRAPH_MODE_OP(reshape_and_cache, ms_custom_ops::CustomReshapeAndCacheOpFuncImpl,
                  ms_custom_ops::CustomReshapeAndCache);

// =============================================================================
// PYBOOST调用实现
// =============================================================================

#include "internal_pyboost_runner.h"

namespace ms_custom_ops {
// 1. 创建算子Pyboost执行器
class ReshapeAndCacheRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetHeadNum(const int32_t &head_num) { this->head_num_ = head_num; }
  void SetCacheMode(const int32_t &cache_mode) { this->cache_mode_ = cache_mode; }

protected:
  internal::InternalOpPtr CreateKernel(
      const internal::InputsImmutableInfoList &inputs,
      const internal::OutputsImmutableInfoList &outputs) override {
    internal::ReshapeAndCacheParam param;
    param.head_num = this->head_num_;
    
    // 根据cache_mode设置格式
    if (this->cache_mode_ == 1) { // NZ格式
      auto inputs_clone = inputs;
      inputs_clone[2].SetFormat(internal::kFormatFRACTAL_NZ);
      inputs_clone[3].SetFormat(internal::kFormatFRACTAL_NZ);
      return internal::CreateAsdReshapeAndCacheOp(inputs_clone, outputs, param,
                                                  internal::kInternalAsdReshapeAndCacheOpName);
    }
    return internal::CreateAsdReshapeAndCacheOp(inputs, outputs, param, 
                                                internal::kInternalAsdReshapeAndCacheOpName);
  }

private:
  int32_t head_num_{0};
  int32_t cache_mode_{0};
};

// 2. 算子kernel调用函数
void npu_reshape_and_cache(const ms::Tensor &key,
                           const std::optional<ms::Tensor> &value,
                           const std::optional<ms::Tensor> &key_cache,
                           const std::optional<ms::Tensor> &value_cache,
                           const std::optional<ms::Tensor> &slot_mapping,
                           std::optional<int64_t> cache_mode,
                           std::optional<int64_t> head_num) {
  auto op_name = "ReshapeAndCache";
  auto runner = std::make_shared<ms_custom_ops::ReshapeAndCacheRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // 设置参数
  if (cache_mode.has_value()) {
    runner->SetCacheMode(static_cast<int32_t>(cache_mode.value()));
  }
  if (head_num.has_value()) {
    runner->SetHeadNum(static_cast<int32_t>(head_num.value()));
  }

  // 执行算子
  runner->Setup(op_name, key, value, key_cache, value_cache, slot_mapping, 
                cache_mode, head_num);
  std::vector<ms::Tensor> inputs = {
      key, GetTensorOrEmpty(value), GetTensorOrEmpty(key_cache),
      GetTensorOrEmpty(value_cache), GetTensorOrEmpty(slot_mapping)};
  std::vector<ms::Tensor> outputs = {};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
}

// 3. pybind接口注册
auto pyboost_reshape_and_cache(const ms::Tensor &key,
                               const std::optional<ms::Tensor> &value,
                               const std::optional<ms::Tensor> &key_cache,
                               const std::optional<ms::Tensor> &value_cache,
                               const std::optional<ms::Tensor> &slot_mapping,
                               std::optional<int64_t> cache_mode,
                               std::optional<int64_t> head_num) {
  // Call<输出Tensor的个数>(算子kernel调用函数, 输入Tensor...)
  return ms::pynative::PyboostRunner::Call<0>(ms_custom_ops::npu_reshape_and_cache, 
                                             key, value, key_cache, value_cache,
                                             slot_mapping, cache_mode, head_num);
}
} // namespace ms_custom_ops

// 注册Python接口
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache", &pyboost_reshape_and_cache, "Reshape And Cache",
        pybind11::arg("key"),
        pybind11::arg("value") = std::nullopt,
        pybind11::arg("key_cache") = std::nullopt,
        pybind11::arg("value_cache") = std::nullopt,
        pybind11::arg("slot_mapping") = std::nullopt,
        pybind11::arg("cache_mode") = std::nullopt,
        pybind11::arg("head_num") = std::nullopt);
}
```

#### 3. 特殊format的支持

**背景说明**：
某些算子需要支持特殊的数据格式（如FRACTAL_NZ），但MindSpore框架不提供自动format推导能力。因此需要通过用户参数来指定格式类型，并配合`trans_data`算子进行格式转换。

**核心概念**：

1. **格式转换算子**：`trans_data`
   - `transdata_type=0`: FRACTAL_NZ_TO_ND (NZ→ND)
   - `transdata_type=1`: ND_TO_FRACTAL_NZ (ND→NZ)
   - 用于在不同数据格式间进行无损转换

2. **算子格式适配**：通过参数控制内部格式处理
   - `cache_mode=0`: ND格式模式（默认）
   - `cache_mode=1`: FRACTAL_NZ格式模式

**典型使用模式**：

**模式1：支持多格式的算子**
```python
# ND格式模式（默认）
ms_custom_ops.reshape_and_cache(key, value, key_cache, value_cache, 
                                slot_mapping, cache_mode=0)

# FRACTAL_NZ格式模式
# 1. 将ND格式缓存转换为NZ格式
key_cache_nz = ms_custom_ops.trans_data(key_cache, transdata_type=1)  # ND→NZ
value_cache_nz = ms_custom_ops.trans_data(value_cache, transdata_type=1)  # ND→NZ

# 2. 使用NZ格式模式执行算子
ms_custom_ops.reshape_and_cache(key, value, key_cache_nz, value_cache_nz, 
                                slot_mapping, cache_mode=1)

# 3. 如需要，将结果转换回ND格式进行验证
key_cache_result = ms_custom_ops.trans_data(key_cache_nz, transdata_type=0)  # NZ→ND
value_cache_result = ms_custom_ops.trans_data(value_cache_nz, transdata_type=0)  # NZ→ND
```

**模式2：专用格式转换算子**
```python
# 单纯的格式转换
nz_tensor = ms_custom_ops.trans_data(nd_tensor, transdata_type=1)  # ND→NZ
nd_tensor = ms_custom_ops.trans_data(nz_tensor, transdata_type=0)   # NZ→ND
```

**实现步骤**：

1. **添加格式选择参数**
   - 为算子添加format选择参数（如`cache_mode`）
   - 定义格式映射关系：`0`=ND格式，`1`=FRACTAL_NZ格式

2. **实现格式转换逻辑**
   - 在`CreateKernel`函数中根据参数值判断是否需要格式转换
   - 对需要特殊格式的输入张量调用`SetFormat()`方法

**代码示例**（以reshape_and_cache为例）：
```cpp
// 在CreateKernel函数中实现格式适配
internal::InternalOpPtr CreateKernel(
    const internal::InputsImmutableInfoList &inputs,
    const internal::OutputsImmutableInfoList &outputs,
    const std::vector<KernelTensor *> &ms_inputs,
    const std::vector<KernelTensor *> &ms_outputs) override {
  
  // 获取格式参数
  auto cache_mode = ms_inputs.at(5); // cache_mode参数位置
  int32_t cache_mode_val = static_cast<int32_t>(cache_mode->GetValue<int64_t>().value());
  
  // 根据参数设置特殊格式
  if (cache_mode_val == 1) { // FRACTAL_NZ格式
    auto inputs_clone = inputs;
    inputs_clone[2].SetFormat(internal::kFormatFRACTAL_NZ); // key_cache
    inputs_clone[3].SetFormat(internal::kFormatFRACTAL_NZ); // value_cache
    return internal::CreateAsdReshapeAndCacheOp(inputs_clone, outputs, param, op_name);
  }
  
  // 默认ND格式，无需转换
  return internal::CreateAsdReshapeAndCacheOp(inputs, outputs, param, op_name);
}
```

**测试中的使用模式**（以NZ格式测试为例）：
```cpp
// NZ Format Test Flow:
// 1. Create initial ND format cache tensors
np_k, np_v, np_k_cache, np_v_cache, np_slot_map = create_nd_inputs(...)

// 2. Convert cache tensors to FRACTAL_NZ format
ms_k_cache = ms_custom_ops.trans_data(ms_k_cache, transdata_type=1)  # ND→NZ
ms_v_cache = ms_custom_ops.trans_data(ms_v_cache, transdata_type=1)  # ND→NZ

// 3. Run ReshapeAndCache with cache_mode=1 (NZ format mode)
net(key, value, ms_k_cache, ms_v_cache, slot_mapping, cache_mode=1)

// 4. Convert results back to ND format for verification
ms_k_cache_nd = ms_custom_ops.trans_data(ms_k_cache, transdata_type=0)  # NZ→ND
ms_v_cache_nd = ms_custom_ops.trans_data(ms_v_cache, transdata_type=0)  # NZ→ND

// 5. Compare with golden ND results
verify_results(ms_k_cache_nd, golden_k_output, dtype)
```

**关键注意事项**：
- ✅ **数据一致性**：格式转换应保持数据完全一致，任何精度损失都可能表明实现错误
- ✅ **Internal算子**：底层算子库会自动处理shape转换，用户只需设置format即可
- ⚠️ **AscendC算子**：需要用户手动实现format转换和shape计算逻辑
- 📝 **参数设计**：建议使用枚举值（0,1,2...）而非字符串，提高性能
- 🔍 **测试验证**：确保不同format下的输入输出shape和数据正确性
- 💡 **性能优化**：避免不必要的格式转换，尽量在同一格式下完成整个计算流程

**格式转换数据类型支持**：
- ✅ **FRACTAL_NZ_TO_ND**: 支持 float16, bfloat16（int8不支持）
- ✅ **ND_TO_FRACTAL_NZ**: 支持 float16, bfloat16, int8
- ⚠️ **对齐要求**: float16/bfloat16需要16字节对齐，int8需要32字节对齐

**适配检查清单**：
- [ ] 是否添加了format选择参数？
- [ ] 是否正确使用了trans_data进行格式转换？
- [ ] 是否在两种模式（graph/pyboost）中都实现了格式转换？
- [ ] 是否验证了不同格式下的功能正确性？
- [ ] 是否测试了格式转换的往返一致性？
- [ ] 是否在文档中说明了参数含义和使用方式？