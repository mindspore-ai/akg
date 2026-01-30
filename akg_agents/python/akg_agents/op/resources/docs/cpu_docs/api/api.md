# CPU C++ API 参考手册

本文档提供 CPU C++ 核心 API 的详细参考，包括函数签名、参数说明和使用示例。

## 1. PyTorch C++扩展API

### PYBIND11_MODULE宏
```cpp
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name", &op_function, "Description");
}
```
- **用途**: 注册自定义算子到PyTorch
- **参数**: 模块对象, 算子名, 函数指针, 描述
- **要求**: 必须使用此宏注册算子
- **注意**: `TORCH_EXTENSION_NAME`是预定义宏

### 张量类型检查
```cpp
torch::ScalarType dtype = x.scalar_type();
bool is_float32 = (dtype == torch::kFloat32);
bool is_float64 = (dtype == torch::kFloat64);
bool is_int32 = (dtype == torch::kInt32);
bool is_int64 = (dtype == torch::kInt64);
```
- **用途**: 检查张量数据类型
- **支持类型**: kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64
- **注意**: 建议优先支持float32/float64和int32/int64

### 数据类型转换
```cpp
torch::Tensor input = x.to(torch::kFloat32);  // 转换为float32
torch::Tensor output = result.to(dtype);      // 转换回原类型
```
- **用途**: 统一数据类型处理
- **建议**: 内部计算使用float32，最后转换回原类型

### 张量连续性检查
```cpp
if (!x.is_contiguous()) {
    x = x.contiguous();
}
```
- **用途**: 确保张量内存布局连续
- **重要性**: 避免非连续内存访问问题

## 2. 数据类型指针获取
```cpp
// float32指针
auto x_ptr = input.data_ptr<float>();
auto out_ptr = output.data_ptr<float>();

// float64指针
auto x_ptr = input.data_ptr<double>();
auto out_ptr = output.data_ptr<double>();

// int32指针
auto x_ptr = input.data_ptr<int32_t>();
auto out_ptr = output.data_ptr<int32_t>();

// int64指针
auto x_ptr = input.data_ptr<int64_t>();
auto out_ptr = output.data_ptr<int64_t>();
```
- **用途**: 获取张量数据指针进行直接操作
- **注意**: 建议优先支持float32/float64和int32/int64，确保张量类型匹配

## 3. 张量操作API

### 张量创建
```cpp
torch::Tensor output = torch::zeros_like(input);  // 创建同形状零张量
torch::Tensor output = torch::ones_like(input);   // 创建同形状单位张量
torch::Tensor output = input.clone();             // 克隆张量
```

### 张量属性
```cpp
int64_t numel = input.numel();           // 元素总数
int64_t dim = input.dim();               // 维度数
torch::IntArrayRef shape = input.sizes(); // 形状
torch::Device device = input.device();   // 设备
```

### 张量验证
```cpp
TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");
```

## 4. 数学运算API

### 基础运算
```cpp
// 最大值
float result = std::max(0.0f, x_ptr[i]);
double result = std::max(0.0, x_ptr[i]);

// 绝对值
float result = std::abs(x_ptr[i]);

// 平方根
float result = std::sqrt(x_ptr[i]);
```

### 条件运算
```cpp
// 三元运算符
float result = (x_ptr[i] > threshold) ? x_ptr[i] : 0.0f;

```

## 5. 类型转换API

### 显式类型转换
```cpp
// float32转float64
double val = static_cast<double>(float_val);

// int32转float32
float val = static_cast<float>(int_val);

// float32转int32
int32_t val = static_cast<int32_t>(float_val);
```

### 数据类型判断
```cpp
bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
                    dtype != torch::kInt32 && dtype != torch::kInt64);
```