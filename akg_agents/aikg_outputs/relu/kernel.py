import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# C++ kernel代码 - 使用AVX2+FMA指令集优化
cpp_source = """
#include <torch/extension.h>
#include <immintrin.h>  // AVX2/FMA intrinsics
#include <cmath>

// 高性能ReLU内核实现
inline void relu_kernel_impl(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N
) {
    // AVX2向量化实现（256位，单次处理8个float32）
    constexpr int VECTOR_SIZE = 8;
    
    int64_t i = 0;
    
    // 向量化主循环
    for (; i + VECTOR_SIZE <= N; i += VECTOR_SIZE) {
        // 加载输入数据
        __m256 vec = _mm256_loadu_ps(&input[i]);
        
        // ReLU计算：max(0, x)
        __m256 zero_vec = _mm256_setzero_ps();
        __m256 result = _mm256_max_ps(vec, zero_vec);
        
        // 存储结果
        _mm256_storeu_ps(&output[i], result);
    }
    
    // 处理剩余元素（标量fallback）
    for (; i < N; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// 多数据类型支持的ReLU内核
inline void relu_kernel_impl_dtype(
    const void* __restrict__ input,
    void* __restrict__ output,
    int64_t N,
    torch::ScalarType dtype
) {
    switch (dtype) {
        case torch::kFloat32: {
            relu_kernel_impl(
                reinterpret_cast<const float*>(input),
                reinterpret_cast<float*>(output),
                N
            );
            break;
        }
        case torch::kFloat64: {
            // double类型处理（AVX2单次处理4个double）
            constexpr int VECTOR_SIZE = 4;
            int64_t i = 0;
            
            const double* in_ptr = reinterpret_cast<const double*>(input);
            double* out_ptr = reinterpret_cast<double*>(output);
            
            for (; i + VECTOR_SIZE <= N; i += VECTOR_SIZE) {
                __m256d vec = _mm256_loadu_pd(&in_ptr[i]);
                __m256d zero_vec = _mm256_setzero_pd();
                __m256d result = _mm256_max_pd(vec, zero_vec);
                _mm256_storeu_pd(&out_ptr[i], result);
            }
            
            for (; i < N; i++) {
                out_ptr[i] = std::max(0.0, in_ptr[i]);
            }
            break;
        }
        case torch::kInt32: {
            // int32类型处理
            const int32_t* in_ptr = reinterpret_cast<const int32_t*>(input);
            int32_t* out_ptr = reinterpret_cast<int32_t*>(output);
            
            for (int64_t i = 0; i < N; i++) {
                out_ptr[i] = std::max(0, in_ptr[i]);
            }
            break;
        }
        case torch::kInt64: {
            // int64类型处理
            const int64_t* in_ptr = reinterpret_cast<const int64_t*>(input);
            int64_t* out_ptr = reinterpret_cast<int64_t*>(output);
            
            for (int64_t i = 0; i < N; i++) {
                out_ptr[i] = std::max(static_cast<int64_t>(0), in_ptr[i]);
            }
            break;
        }
        default:
            TORCH_CHECK(false, "Unsupported data type for ReLU");
    }
}

// PyTorch接口函数
torch::Tensor relu_kernel(torch::Tensor x) {
    // 1. 检查输入有效性
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
    
    // 2. 确保张量连续
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    
    // 3. 检查数据类型
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64 && 
                        dtype != torch::kInt32 && dtype != torch::kInt64);
    
    // 4. 类型转换（不支持的类型转为float32）
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    
    // 5. 创建输出张量
    torch::Tensor output = torch::empty_like(input);
    
    // 6. 调用高性能内核
    relu_kernel_impl_dtype(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        input.scalar_type()
    );
    
    // 7. 转换回原类型
    if (need_convert) {
        output = output.to(dtype);
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_kernel", &relu_kernel, "CPU ReLU operator with AVX2 optimization");
}
"""

# 加载C++扩展
relu_module = load_inline(
    name="custom_relu",
    cpp_sources=cpp_source,
    extra_cflags=["-O3", "-mavx2", "-mfma"],  # 启用AVX2和FMA优化
    verbose=False
)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # ReLU算子不需要参数
    
    def forward(self, x):
        # 确保输入在CPU上
        if x.device.type != "cpu":
            x = x.cpu()
        
        # 调用C++ kernel
        return relu_module.relu_kernel(x)