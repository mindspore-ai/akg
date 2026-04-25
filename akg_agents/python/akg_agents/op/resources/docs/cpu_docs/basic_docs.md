# CPU C++ 编程基础教程

本文档介绍 CPU C++ 的核心概念和标准编程模式，通过详细示例帮助理解如何构建高效的内核。

## 1. 核心概念

### 内核 (Kernel)
- **定义**: 使用 `PYBIND11_MODULE` 注册的 C++ 函数，编译后在 CPU 上执行
- **特点**: 直接操作张量数据指针，支持多种数据类型

### 张量处理
- **连续性**: 确保张量内存布局连续，避免非连续访问
- **类型统一**: 内部计算使用统一类型，最后转换回原类型
- **边界检查**: 所有数组访问前必须检查边界

### 内存管理
- **自动管理**: PyTorch 自动管理张量内存生命周期
- **指针操作**: 直接操作数据指针进行高效计算
- **类型安全**: 确保指针类型与张量类型匹配

## 2. 标准内核结构

CPU C++ 内核都遵循相同的五步结构模式：

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

    // 4. 根据数据类型手动计算
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
    }

    // 5. 转换回原类型
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}
```

## 3. 编程模式

### 3.1 元素级操作模式
适用于激活函数、逐元素运算等简单操作。

```cpp
torch::Tensor elementwise_kernel(torch::Tensor x) {
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
            out_ptr[i] = std::max(0.0f, x_ptr[i]);  // ReLU: max(0, x)
        }
    }

    if (need_convert) output = output.to(dtype);
    return output;
}
```

### 3.2 归约操作模式
适用于求和、最大值、最小值等聚合操作。

```cpp
torch::Tensor reduction_kernel(torch::Tensor x) {
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
    }
    
    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 4. 边界处理示例

### 张量边界检查
```cpp
torch::Tensor safe_tensor_operation(torch::Tensor x) {
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
            // 边界检查：确保索引有效
            if (i < numel) {
                out_ptr[i] = std::max(0.0f, x_ptr[i]);
            }
        }
    }
    
    if (need_convert) output = output.to(dtype);
    return output;
}
```