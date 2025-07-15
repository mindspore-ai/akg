# ms_custom_ops 自定义算子使用指南

## 概述

ms_custom_ops 提供了一套完整的自定义算子框架，支持在 PyNative 和 Graph 两种执行模式下高效实现自定义算子。本文档将指导您如何使用这套框架开发和集成自定义算子。

## 系统架构

### 核心组件

1. **PyBoost 框架**：用于 PyNative 模式下的动态执行
2. **GraphMode 框架**：用于静态图编译模式
3. **共享组件**：包括内存管理、缓存优化等通用功能

### 目录结构

```
ms_custom_ops/
├── src/
│   ├── ops_def/
│   │   ├── ms_kernels_internal/
│   │   │   ├── pyboost/           # PyNative模式实现
│   │   │   ├── graphmode/         # Graph模式实现
│   │   │   └── 共享组件文件
│   │   └── CMakeLists.txt
│   └── python/
└── tests/
```

## 快速开始

### 1. 环境准备

确保已安装：
- MindSpore >= 2.0
- 昇腾 CANN 工具包
- CMake >= 3.14
- Python >= 3.9

### 2. 安装编译

```bash
# 克隆代码仓库
git clone <repository_url>
cd ms_custom_ops

# 安装（会自动编译自定义算子）
pip install -e .
```

编译过程会自动：
- 检测昇腾环境
- 使用 CMake 构建自定义算子
- 将生成的 .so 文件安装到正确位置

### 3. 使用示例

#### PyNative 模式

```python
import mindspore as ms
from vllm_mindspore import ms_custom_ops

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
from vllm_mindspore import ms_custom_ops

# 设置为 Graph 模式
ms.set_context(mode=ms.context.GRAPH_MODE)
ms.set_device("Ascend")

# 使用 ModuleWrapper 封装
class MyNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        
    def construct(self, key, value, key_cache, value_cache, slot_mapping, head_num):
        mod = ModuleWrapper("custom_reshape_and_cache", ms_custom_ops)
        return mod.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, head_num)

# 使用网络
net = MyNet()
output = net(key, value, key_cache, value_cache, slot_mapping, head_num)
```

## 开发自定义算子

### 1. 创建算子实现

#### PyBoost 模式实现

在 `ms_custom_ops/src/ops_def/ms_kernels_internal/pyboost/ops/` 下创建新文件：

```cpp
// my_op_runner.cc
#include "ms_custom_ops/src/ops_def/ms_kernels_internal/pyboost/internal_pyboost_runner.h"

class MyOpRunner : public InternalPyboostRunner {
public:
    MyOpRunner() : InternalPyboostRunner("my_op", "MyOp") {}
    
    // 实现算子逻辑
    void Setup(const diopiContext_t& pycontent, ...) override {
        // 1. 设置参数
        // 2. 计算 hash key
        // 3. 创建内核
    }
};

// 注册算子
MS_KERNELS_INTERNAL_FACTORY_REG(MyOp, MyOpRunner);
```

#### GraphMode 实现

在 `ms_custom_ops/src/ops_def/ms_kernels_internal/graphmode/ops/` 下创建新文件：

```cpp
// my_op.cc
#include "ms_custom_ops/src/ops_def/ms_kernels_internal/graphmode/internal_kernel_mod.h"

class CustomMyOp : public InternalKernelMod {
public:
    CustomMyOp() : InternalKernelMod("my_op") {}
    
    bool Init(const PrimitivePtr &primitive, ...) override {
        // 初始化参数
    }
    
    bool Launch(const std::vector<KernelTensor*> &inputs, ...) override {
        // 执行算子逻辑
    }
};

// 注册算子
MS_CUSTOM_INTERNAL_KERNEL_FACTORY_REG(MyOp, CustomMyOp);
```

### 2. 添加 Python 接口

在 `vllm_mindspore/ms_custom_ops/__init__.py` 中添加：

```python
def my_op(*args, **kwargs):
    """My custom operator"""
    return ops.Custom(func_type="internal", func_name="MyOp", out_shape=..., out_dtype=...)(*args, **kwargs)
```

### 3. 编写测试

创建测试文件 `ms_custom_ops/tests/test_my_op.py`：

```python
import pytest
import numpy as np
import mindspore as ms
from vllm_mindspore import ms_custom_ops

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

### 1. Hash 缓存优化

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

**Q: 编译失败提示找不到 CANN 环境**  
A: 确保正确安装昇腾 CANN 工具包，并设置环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**Q: 算子在不同模式下行为不一致**  
A: 检查是否正确处理了 Parameter 和 Tensor 的区别，Graph 模式下缓存通常使用 Parameter。

**Q: 性能不如预期**  
A: 1) 检查是否正确使用了缓存机制；2) 确认内存访问模式是否高效；3) 使用 Profiler 定位瓶颈。

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

## 许可证

本项目采用 Apache License 2.0 许可证。