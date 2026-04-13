---
name: cpu-optimization-x64
description: "x64 CPU 架构性能优化技巧、SIMD/AVX 向量化、数值稳定性和调试策略"
category: method
version: "1.0.0"
metadata:
  backend: cpu
  dsl: cpp
  architecture: x86_64
  optimization_techniques: "SIMD, AVX, AVX2, AVX-512, cache optimization, loop unrolling"
---

# x64 CPU 性能优化指南

## 1. x64 架构特性与优化策略

### 1.1 架构标识

- **架构**: x86_64 (也称为 x64, AMD64)
- **主要厂商**: Intel, AMD
- **SIMD 扩展**: AVX, AVX2, AVX-512

### 1.2 核心优化原则

1. **利用 SIMD 并行性**: 使用 AVX/AVX2/AVX-512 指令同时处理多个数据
2. **优化缓存使用**: 按行优先访问，提高缓存命中率
3. **减少分支预测失败**: 循环展开，减少条件判断
4. **内存对齐**: 确保数据对齐到 32/64 字节边界

## 2. SIMD/AVX 向量化优化

### 2.1 基本概念

**AVX (Advanced Vector Extensions)** 是 x86-64 的 SIMD 指令集扩展：

- **AVX**: 256 位寄存器，可同时处理 8 个 float32 或 4 个 float64
- **AVX2**: 增强的 AVX，支持整数运算
- **AVX-512**: 512 位寄存器，可同时处理 16 个 float32 或 8 个 float64

### 2.2 编译器自动向量化

**推荐方式**: 让编译器自动向量化，通过编译选项启用：

```python
# 在 load_inline 中添加向量化选项
op_module = load_inline(
    name="custom_op",
    cpp_sources=cpp_source,
    extra_cflags=[
        "-O3",              # 最高优化级别
        "-march=native",    # 针对当前 CPU 架构优化
        "-ftree-vectorize", # 启用自动向量化
    ],
    verbose=True
)
```

### 2.3 循环优化示例

**简单方式**（未优化）:

```cpp
torch::Tensor elementwise_add(torch::Tensor a, torch::Tensor b) {
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();
    
    torch::Tensor output = torch::zeros_like(a);
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = a.numel();
    
    // 简单循环
    for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    
    return output;
}
```

**优化方式**（循环展开，便于向量化）:

```cpp
torch::Tensor elementwise_add_optimized(torch::Tensor a, torch::Tensor b) {
    if (!a.is_contiguous()) a = a.contiguous();
    if (!b.is_contiguous()) b = b.contiguous();
    
    torch::Tensor output = torch::zeros_like(a);
    auto a_ptr = a.data_ptr<float>();
    auto b_ptr = b.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = a.numel();
    
    // 循环展开 8 倍（匹配 AVX 寄存器宽度）
    int64_t i = 0;
    int64_t step = 8;
    for (; i + step <= numel; i += step) {
        out_ptr[i]     = a_ptr[i]     + b_ptr[i];
        out_ptr[i + 1] = a_ptr[i + 1] + b_ptr[i + 1];
        out_ptr[i + 2] = a_ptr[i + 2] + b_ptr[i + 2];
        out_ptr[i + 3] = a_ptr[i + 3] + b_ptr[i + 3];
        out_ptr[i + 4] = a_ptr[i + 4] + b_ptr[i + 4];
        out_ptr[i + 5] = a_ptr[i + 5] + b_ptr[i + 5];
        out_ptr[i + 6] = a_ptr[i + 6] + b_ptr[i + 6];
        out_ptr[i + 7] = a_ptr[i + 7] + b_ptr[i + 7];
    }
    
    // 处理剩余元素
    for (; i < numel; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
    
    return output;
}
```

**优化效果**: 循环展开后，编译器更容易识别并生成 AVX 向量化指令，性能提升 4-8 倍。

### 2.4 Reduction 操作优化

**简单方式**:

```cpp
float sum_simple(const float* data, int64_t size) {
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}
```

**优化方式**（分块累加）:

```cpp
float sum_optimized(const float* data, int64_t size) {
    // 使用 8 个累加器，减少数据依赖
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
    
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        sum0 += data[i];
        sum1 += data[i + 1];
        sum2 += data[i + 2];
        sum3 += data[i + 3];
        sum4 += data[i + 4];
        sum5 += data[i + 5];
        sum6 += data[i + 6];
        sum7 += data[i + 7];
    }
    
    // 合并结果
    float sum = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
    
    // 处理剩余元素
    for (; i < size; ++i) {
        sum += data[i];
    }
    
    return sum;
}
```

**关键优化**: 使用多个累加器避免循环携带依赖，允许指令级并行和向量化。

## 3. 缓存优化

### 3.1 缓存层次

- **L1 Cache**: 32-64 KB，延迟 ~4 周期
- **L2 Cache**: 256-512 KB，延迟 ~12 周期
- **L3 Cache**: 8-32 MB（共享），延迟 ~40 周期
- **主内存**: 延迟 ~200 周期

### 3.2 优化策略

**原则**: 按行优先访问，提高空间局部性

```cpp
// 二维矩阵转置优化示例
torch::Tensor transpose_optimized(torch::Tensor input) {
    if (!input.is_contiguous()) input = input.contiguous();
    
    auto sizes = input.sizes();
    int64_t M = sizes[0];
    int64_t N = sizes[1];
    
    torch::Tensor output = torch::zeros({N, M}, input.options());
    auto in_ptr = input.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    
    // 分块处理，提高缓存命中率
    const int64_t BLOCK_SIZE = 64;  // 适配缓存行大小
    
    for (int64_t i = 0; i < M; i += BLOCK_SIZE) {
        for (int64_t j = 0; j < N; j += BLOCK_SIZE) {
            int64_t i_max = std::min(i + BLOCK_SIZE, M);
            int64_t j_max = std::min(j + BLOCK_SIZE, N);
            
            for (int64_t ii = i; ii < i_max; ++ii) {
                for (int64_t jj = j; jj < j_max; ++jj) {
                    out_ptr[jj * M + ii] = in_ptr[ii * N + jj];
                }
            }
        }
    }
    
    return output;
}
```

## 4. 数值稳定性优化

### 4.1 防止 Softmax 溢出

```cpp
torch::Tensor softmax_stable(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    
    torch::Tensor output = torch::zeros_like(x);
    auto x_ptr = x.data_ptr<float>();
    auto out_ptr = output.data_ptr<float>();
    int64_t numel = x.numel();
    
    // 找到最大值（防止 exp 溢出）
    float max_val = x_ptr[0];
    for (int64_t i = 1; i < numel; ++i) {
        max_val = std::max(max_val, x_ptr[i]);
    }
    
    // 减去最大值后计算 exp
    float sum = 0.0f;
    for (int64_t i = 0; i < numel; ++i) {
        float exp_val = std::exp(x_ptr[i] - max_val);
        out_ptr[i] = exp_val;
        sum += exp_val;
    }
    
    // 归一化
    for (int64_t i = 0; i < numel; ++i) {
        out_ptr[i] /= sum;
    }
    
    return output;
}
```

### 4.2 Kahan 求和算法（提升精度）

```cpp
float kahan_sum(const float* data, int64_t size) {
    float sum = 0.0f;
    float c = 0.0f;  // 补偿变量
    
    for (int64_t i = 0; i < size; ++i) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    
    return sum;
}
```

**使用场景**: 处理大量浮点数累加时，减少精度损失。

## 5. 完整优化示例：ReLU

```cpp
torch::Tensor relu_optimized(torch::Tensor x) {
    // 1. 确保连续性
    if (!x.is_contiguous()) x = x.contiguous();
    
    // 2. 类型检查与转换
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kFloat64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    
    // 3. 创建输出
    torch::Tensor output = torch::zeros_like(input);
    
    // 4. 优化的计算逻辑
    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        
        // 循环展开 8 倍
        int64_t i = 0;
        for (; i + 8 <= numel; i += 8) {
            out_ptr[i]     = std::max(0.0f, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0f, x_ptr[i + 1]);
            out_ptr[i + 2] = std::max(0.0f, x_ptr[i + 2]);
            out_ptr[i + 3] = std::max(0.0f, x_ptr[i + 3]);
            out_ptr[i + 4] = std::max(0.0f, x_ptr[i + 4]);
            out_ptr[i + 5] = std::max(0.0f, x_ptr[i + 5]);
            out_ptr[i + 6] = std::max(0.0f, x_ptr[i + 6]);
            out_ptr[i + 7] = std::max(0.0f, x_ptr[i + 7]);
        }
        
        // 处理剩余元素
        for (; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kFloat64) {
        auto x_ptr = input.data_ptr<double>();
        auto out_ptr = output.data_ptr<double>();
        int64_t numel = input.numel();
        
        // 同样的循环展开
        int64_t i = 0;
        for (; i + 4 <= numel; i += 4) {  // double 展开 4 倍
            out_ptr[i]     = std::max(0.0, x_ptr[i]);
            out_ptr[i + 1] = std::max(0.0, x_ptr[i + 1]);
            out_ptr[i + 2] = std::max(0.0, x_ptr[i + 2]);
            out_ptr[i + 3] = std::max(0.0, x_ptr[i + 3]);
        }
        
        for (; i < numel; ++i) {
            out_ptr[i] = std::max(0.0, x_ptr[i]);
        }
    }
    
    // 5. 类型还原
    if (need_convert) output = output.to(dtype);
    return output;
}
```

## 6. 性能调试与分析

### 6.1 性能检查清单

- [ ] 是否启用了 `-O3` 优化？
- [ ] 是否添加了 `-march=native`？
- [ ] 循环是否展开（8 倍 for float32, 4 倍 for float64）？
- [ ] 是否按行优先访问内存？
- [ ] 是否避免了不必要的类型转换？
- [ ] Reduction 操作是否使用了多累加器？

### 6.2 编译选项建议

```python
extra_cflags = [
    "-O3",                  # 最高优化级别
    "-march=native",        # 针对当前 CPU
    "-ftree-vectorize",     # 自动向量化
    "-ffast-math",          # 快速数学（牺牲部分精度）
    "-funroll-loops",       # 循环展开
]
```

**注意**: `-ffast-math` 可能影响数值精度，谨慎使用。

## 7. 常见优化误区

| 误区 | 说明 | 建议 |
|------|------|------|
| 过度手动向量化 | 手写 AVX intrinsics 代码复杂且易错 | 优先让编译器自动向量化 |
| 循环展开太多 | 过度展开增加代码体积，降低 I-Cache 命中率 | Float32 展开 8 倍，Float64 展开 4 倍 |
| 忽略数据对齐 | 未对齐访问降低性能 | 使用 `torch::zeros_like` 等自动对齐 |
| 不合理的精度提升 | 内部计算无需强制使用 double | Float32 已足够，避免不必要转换 |

## 8. 总结

### x64 优化关键原则

1. **编译器自动向量化**: 使用 `-O3 -march=native -ftree-vectorize`
2. **循环展开**: Float32 展开 8 倍，Float64 展开 4 倍
3. **多累加器**: Reduction 操作使用多个累加器避免依赖
4. **缓存友好**: 按行优先访问，大矩阵分块处理
5. **数值稳定**: Softmax 减去最大值，大量累加使用 Kahan 算法

### 参考资料

- Intel 优化手册: https://www.intel.com/content/www/us/en/developer/articles/technical/improve-performance-with-vectorization.html
- AVX 内在函数参考: https://www.intel.com/content/www/us/en/docs/cpp-compiler/developer-guide-reference/2021-8/details-of-avx-intrinsics.html
