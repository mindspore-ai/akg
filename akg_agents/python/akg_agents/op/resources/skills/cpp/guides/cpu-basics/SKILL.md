---
name: cpu-basics
description: "CPU C++ 算子核心概念、标准结构模式、KernelBench 代码规范和内嵌扩展方法"
category: fundamental
version: "1.0.0"
metadata:
  backend: cpu
  dsl: cpp
  operator_patterns: "all"
  architecture: "x86_64, aarch64"
---

# CPU C++ 编程基础

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `PYBIND11_MODULE` 注册的 C++ 函数，编译后在 CPU 上执行
- **特点**: 直接操作张量数据指针，支持多种数据类型
- **形式**: 使用 PyTorch C++ 扩展，通过 `load_inline` 动态编译加载

### 张量处理
- **连续性**: 确保张量内存布局连续，避免非连续访问
- **类型统一**: 内部计算使用统一类型（优先 float32/float64/int32/int64），最后转换回原类型
- **边界检查**: 所有数组访问前必须检查边界

### 内存管理
- **自动管理**: PyTorch 自动管理张量内存生命周期
- **指针操作**: 直接操作数据指针进行高效计算
- **类型安全**: 确保指针类型与张量类型匹配

## 2. 标准内核结构（五步模式）

所有 CPU C++ 内核都遵循相同的五步结构模式：

```cpp
torch::Tensor standard_kernel(torch::Tensor x) {
    // 1. 确保输入张量是连续的
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // 2. 检查数据类型，支持多种类型
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
                        dtype != torch::kInt32 && dtype != torch::kInt64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // 3. 创建输出张量
    torch::Tensor output = torch::zeros_like(input);

    // 4. 根据数据类型分发计算
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);  // ReLU: max(0, x)
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt32) {
        auto x_ptr = input.data_ptr<int32_t>();
        auto out_ptr = output.data_ptr<int32_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt64) {
        auto x_ptr = input.data_ptr<int64_t>();
        auto out_ptr = output.data_ptr<int64_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0L, x_ptr[i]);
        }
    }

    // 5. 转换回原类型
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}
```

## 3. KernelBench 标准代码格式

**重要**: 生成的代码必须遵循 KernelBench 格式规范，使用 **Python 模块内嵌 C++ 代码** 的方式。

### 完整模板示例

参考示例位置: `akg_agents/python/akg_agents/op/resources/docs/cpu_docs/examples/torch_xxx_kernel.py`

```python
import torch
from torch.utils.cpp_extension import load_inline

# 内联C++扩展代码
cpp_source = """
#include <torch/extension.h>

torch::Tensor op_name_kernel(torch::Tensor x) {
    // 1. 确保输入张量是连续的
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // 2. 检查数据类型，支持多种类型
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
                        dtype != torch::kInt32 && dtype != torch::kInt64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;

    // 3. 创建输出张量
    torch::Tensor output = torch::zeros_like(input);

    // 4. 根据数据类型分发计算
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            // 具体的算子计算逻辑
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        // 同样的逻辑，但使用 double 类型
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt32) {
        auto x_ptr = input.data_ptr<int32_t>();
        auto out_ptr = output.data_ptr<int32_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt64) {
        auto x_ptr = input.data_ptr<int64_t>();
        auto out_ptr = output.data_ptr<int64_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = compute_logic(x_ptr[i]);
        }
    }

    // 5. 转换回原类型
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name_kernel", &op_name_kernel, "CPU op_name operator");
}
"""

# 动态加载C++扩展
op_name_module = load_inline(
    name="custom_op_name",
    cpp_sources=cpp_source,
    extra_cflags=["-O3"],
    verbose=True
)

# Python接口函数
def op_name(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        x = x.cpu()
    return op_name_module.op_name_kernel(x)
```

### 关键要点

1. **内嵌 C++ 代码**: 使用三引号字符串包含完整的 C++ 源码
2. **动态编译**: 使用 `load_inline` 动态编译并加载扩展
3. **PYBIND11 注册**: 必须使用 `PYBIND11_MODULE` 宏注册算子
4. **Python 接口**: 提供简洁的 Python 函数包装
5. **不包含测试代码**: 生成的代码中不要包含任何测试代码

## 4. 三种基本编程模式

### 4.1 元素级操作模式

适用于激活函数、逐元素运算等简单操作。

```cpp
// ReLU: max(0, x)
torch::Tensor relu_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    torch::Tensor output = torch::zeros_like(input);

    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 4.2 归约操作模式

适用于求和、最大值、最小值等聚合操作。

```cpp
// Sum reduction: 沿指定维度求和
torch::Tensor sum_reduction_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    
    int64_t numel = input.numel();
    torch::Tensor output;
    
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        float result = 0.0f;
        for (int64_t i = 0; i < numel; ++i) {
            result += x_ptr[i];  // 求和归约
        }
        output = torch::tensor({result}, torch::kFloat32);
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        double result = 0.0;
        for (int64_t i = 0; i < numel; ++i) {
            result += x_ptr[i];
        }
        output = torch::tensor({result}, torch::kFloat64);
    }
    
    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 4.3 边界安全处理模式

确保所有操作都有适当的边界检查和错误处理。

```cpp
torch::Tensor safe_operation_kernel(torch::Tensor x) {
    // 1. 检查张量有效性
    TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
    TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
    
    // 2. 确保张量连续性
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // 3. 类型检查和转换
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    torch::Tensor output = torch::zeros_like(input);
    
    // 4. 安全的数据处理
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }
    
    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 5. 核心 API 参考

### 张量类型检查与转换

```cpp
// 类型检查
torch::ScalarType dtype = x.scalar_type();
bool is_float32 = (dtype == torch::kFloat32);
bool is_float64 = (dtype == torch::kFloat64);
bool is_int32 = (dtype == torch::kInt32);
bool is_int64 = (dtype == torch::kInt64);

// 类型转换
torch::Tensor input = x.to(torch::kFloat32);  // 转换为 float32
torch::Tensor output = result.to(dtype);      // 转换回原类型
```

### 连续性检查

```cpp
if (!x.is_contiguous()) {
    x = x.contiguous();
}
```

### 数据指针获取

```cpp
// float32 指针
auto x_ptr = input.data_ptr<float>();
auto out_ptr = output.data_ptr<float>();

// float64 指针
auto x_ptr = input.data_ptr<double>();
auto out_ptr = output.data_ptr<double>();

// int32 指针
auto x_ptr = input.data_ptr<int32_t>();
auto out_ptr = output.data_ptr<int32_t>();

// int64 指针
auto x_ptr = input.data_ptr<int64_t>();
auto out_ptr = output.data_ptr<int64_t>();
```

### 张量创建与属性

```cpp
// 创建输出张量
torch::Tensor output = torch::zeros_like(input);  // 同形状零张量
torch::Tensor output = torch::ones_like(input);   // 同形状单位张量
torch::Tensor output = input.clone();             // 克隆张量

// 张量属性
int64_t numel = input.numel();           // 元素总数
int64_t dim = input.dim();               // 维度数
torch::IntArrayRef shape = input.sizes(); // 形状
```

### 边界检查

```cpp
TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
```

## 6. 编程约束与最佳实践

### 必须遵循的规则

1. **边界检查**: 所有数组访问前必须检查边界
2. **类型安全**: 确保指针类型与张量类型匹配
3. **连续性保证**: 处理前确保张量内存连续
4. **类型支持**: 优先支持 float32/float64/int32/int64，其他类型自动转换

### 内核设计原则

1. **单一职责**: 每个函数只做一件事
2. **参数简单**: 避免复杂的数据结构传递
3. **避免动态分配**: 内核内避免 new/delete
4. **清晰注释**: 添加充分的注释说明计算逻辑

### OpenMP并行编程约束

1. **⚠️ 关键约束**: OpenMP运行时API调用位置限制
   - **禁止场景**: 不得在SIMD区域、并行区域的intervening code中调用`omp_get_thread_num()`等OpenMP运行时API
   - **正确用法**: OpenMP API应在并行区域内的正常代码路径中调用,而非在编译器受限的上下文中
   - **错误示例**:
     ```cpp
     // ❌ 错误:在受限上下文中调用OpenMP API
     std::mt19937 gen(seed + omp_get_thread_num());  // 编译错误!
     ```
   - **正确示例**:
     ```cpp
     // ✅ 正确:在并行区域内正常调用
     #pragma omp parallel
     {
         int tid = omp_get_thread_num();  // 正确
         std::mt19937 gen(seed + tid);
     }
     ```
2. **线程安全**: 确保每个线程有独立的随机数生成器实例
3. **数据竞争**: 避免多个线程同时写入同一内存位置

### 代码风格要求

1. **不包含测试代码**: 生成的代码不要包含任何测试代码
2. **内嵌 C++ 格式**: C++ 代码必须写在三引号字符串中
3. **保持格式清晰**: 适当的缩进和换行
4. **描述性命名**: 使用清晰的变量名和函数名

## 7. 更多示例参考

更多完整的算子实现示例，请参考：

- **基础文档**: `akg_agents/python/akg_agents/op/resources/docs/cpu_docs/basic_docs.md`
- **优化建议**: `akg_agents/python/akg_agents/op/resources/docs/cpu_docs/suggestion_docs.md`
- **API 手册**: `akg_agents/python/akg_agents/op/resources/docs/cpu_docs/api/api.md`
- **代码模板**: `akg_agents/python/akg_agents/op/resources/docs/cpu_docs/examples/torch_xxx_kernel.py`

这些文档提供了完整的实现指南和参考模板。
