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
│   |   └── {op_name}_doc.yaml
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

// 注册算子infer函数，用于在计算过程中推导算子输出shape和dtype，以便分配算子输出内存
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
#include "internal_kernel_mod.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ms_extension/api.h"
#include "ops/base_operator.h"
#include "ops/ops_func_impl/op_func_impl.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "runtime/device/kernel_runtime.h"
#include "utils/check_convert_utils.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// 图模式调用实现
// =============================================================================

namespace ms_custom_ops {
// 算子infer函数，需要实现InferShape和InferType函数
class OPS_API CustomReshapeAndCacheOpFuncImpl : public OpFuncImpl {
public:
  // 算子infershape，需要返回算子所有输出的shape大小
  ShapeArray InferShape(const PrimitivePtr &primitive,
                        const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetShape()};
  }

  // 算子infertype，需要返回算子所有输出的数据类型
  std::vector<TypeId> InferType(const PrimitivePtr &primitive,
                                const InferInfoPtrList &input_infos) const override {
    return {input_infos[0]->GetType()};
  }

  bool GeneralInferRegistered() const override { return true; }
};

constexpr size_t kInputKeyIndex = 0;
constexpr size_t kInputValueIndex = 1;
constexpr size_t kInputKeyCacheIndex = 2;
constexpr size_t kInputValueCacheIndex = 3;
constexpr size_t kInputSlotMappingIndex = 4;
constexpr size_t kInputHeadNumIndex = 5;
constexpr size_t kOutputIndex = 0;
// 算子graph模式调用，需要继承InternalKernelMod基类，并实现InitKernelInputsOutputsIndex和CreateKernel函数
class CustomReshapeAndCache : public InternalKernelMod {
public:
  CustomReshapeAndCache() : InternalKernelMod() {}
  ~CustomReshapeAndCache() = default;

  // 是算子前端定义的输入输出和算子kernel输入输出位置索引的映射关系。
  void InitKernelInputsOutputsIndex() override {
    kernel_inputs_index_ = {kInputKeyIndex, kInputValueIndex, kInputKeyCacheIndex,
                            kInputValueCacheIndex, kInputSlotMappingIndex};
    kernel_outputs_index_ = {kOutputIndex};
  }

protected:
  // 创建具体算子的op实例
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override {
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, internal::kInternalReshapeAndCacheOpName);
  }
};
} // namespace ms_custom_ops

// 注册算子infer函数，用于在计算过程中推导算子输出shape和dtype，以便分配算子输出内存
REG_GRAPH_MODE_OP(reshape_and_cache, ms_custom_ops::CustomReshapeAndCacheOpFuncImpl,
                  ms_custom_ops::CustomReshapeAndCache);

// =============================================================================
// PYBOOST调用实现
// =============================================================================

#include "internal_pyboost_runner.h"

using namespace ms_custom_ops;
namespace ms::pynative {

// 创建算子pyboost执行器，需要继承InternalPyboostRunner
class ReshapeAndCacheRunner : public InternalPyboostRunner {
public:
  using InternalPyboostRunner::InternalPyboostRunner;

  void SetHeadNum(const int32_t &head_num) { this->head_num_ = head_num; }

protected:
   // 创建具体算子的op实例
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override {
    return internal::CreateReshapeAndCacheOp(
        inputs, outputs, internal::kInternalReshapeAndCacheOpName);
  }

private:
  int32_t head_num_{0};
};

// 算子注册
MS_KERNELS_INTERNAL_NAME_REG(ReshapeAndCache,
                             internal::kInternalReshapeAndCacheOpName);
} // namespace ms::pynative

namespace ms_custom_ops {
// 获取tensor或创建空tensor
ms::Tensor GetTensorOrEmpty(const std::optional<ms::Tensor> &opt_tensor) {
  return opt_tensor.has_value() ? opt_tensor.value() : ms::Tensor();
}

// 算子kernel调用函数，需要手动创建输出tensor
void npu_reshape_and_cache(const ms::Tensor &key,
                           const std::optional<ms::Tensor> &value,
                           const std::optional<ms::Tensor> &key_cache,
                           const std::optional<ms::Tensor> &value_cache,
                           const std::optional<ms::Tensor> &slot_mapping,
                           std::optional<int64_t> head_num) {
  auto op_name = "ReshapeAndCache";
  auto runner = std::make_shared<ms::pynative::ReshapeAndCacheRunner>(op_name);
  MS_EXCEPTION_IF_NULL(runner);

  // 设置head_num属性
  if (head_num.has_value()) {
    runner->SetHeadNum(static_cast<int32_t>(head_num.value()));
  }

  // 索引入参设置到runner
  runner->Setup(op_name, key, value, key_cache, value_cache, slot_mapping,
                head_num);

  // 获取输入输出tensor;
  std::vector<ms::Tensor> inputs = {
      key, GetTensorOrEmpty(value), GetTensorOrEmpty(key_cache),
      GetTensorOrEmpty(value_cache), GetTensorOrEmpty(slot_mapping)};
  std::vector<ms::Tensor> outputs = {};
  runner->GetOrCreateKernel(inputs, outputs);
  runner->Run(inputs, outputs);
  return;
}
} // namespace ms_custom_ops

// pybind调用函数
auto pyboost_reshape_and_cache(const ms::Tensor &key,
                               const std::optional<ms::Tensor> &value,
                               const std::optional<ms::Tensor> &key_cache,
                               const std::optional<ms::Tensor> &value_cache,
                               const std::optional<ms::Tensor> &slot_mapping,
                               std::optional<int64_t> head_num) {
  return ms::pynative::PyboostRunner::Call<0>(
      ms_custom_ops::npu_reshape_and_cache, key, value, key_cache, value_cache,
      slot_mapping, head_num);
}

// 算子接口注册，对接C++和python接口
MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("reshape_and_cache", &pyboost_reshape_and_cache, "Reshape And Cache",
        pybind11::arg("key"), pybind11::arg("value") = std::nullopt,
        pybind11::arg("key_cache") = std::nullopt,
        pybind11::arg("value_cache") = std::nullopt,
        pybind11::arg("slot_mapping") = std::nullopt,
        pybind11::arg("head_num") = std::nullopt);
}
```

#### 3. 编写测试

创建测试文件 `tests/st/test_my_op.py`：

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


## 🐛 调试技巧

### 1. 日志输出

设置环境变量开启详细日志：
```bash
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
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

## 📋 文件命名规范

为了保持项目结构的一致性，请遵循以下命名规范：

### 算子实现文件
- **算子**: `{op_name}.cc` (如: `reshape_and_cache.cc`)
- **AscendC算子kernel**：按照AscendC官方要求实现`op_host`和`op_kernel`目录下算子文件。

### 配置文件
- **YAML配置**: `{op_name}_op.yaml` (如: `reshape_and_cache_op.yaml`)
- **算子文档**: `{op_name}_doc.yaml` (如: `reshape_and_cache_doc.yaml`)

### 测试文件
- **测试文件**: `test_{op_name}.py` (如: `test_reshape_and_cache.py`)

### 头文件
- **基类头文件**: 使用描述性名称 (如: `internal_pyboost_runner.h`)
- **工具头文件**: 使用功能描述 (如: `internal_helper.h`)

## 🤝 贡献指南

欢迎贡献新的自定义算子！请遵循以下步骤：

1. **Fork** 代码仓库
2. **创建特性分支**: `git checkout -b feature/your-new-op`
3. **实现算子**并添加测试
4. **提交更改**: `git commit -m "Add new operator: your-new-op"`
5. **推送分支**: `git push origin feature/your-new-op`
6. **创建 Pull Request**

确保：
- 代码符合项目编码规范
- 添加充分的单元测试
- 更新相关文档
- 遵循文件命名规范
- 通过所有测试用例

## 📄 许可证

本项目采用 Apache License 2.0 许可证。