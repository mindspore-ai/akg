---
name: cuda-c-optimization
description: "CUDA C 性能优化、数值稳定性和调试排查"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: cuda_c
structure:
  child_skills:
    - cuda-c-patterns
---

# CUDA C 性能优化指南

## 1. 性能优化策略

### 1.1 块大小选择策略

- **基础**: 使用 2 的幂（128, 256, 512, 1024）
- **推荐**: 256 或 512 线程每块
- **限制**: 每块最多 1024 线程（大多数 GPU）
- **调优**: 平衡并行度与资源占用，避免过大或过小

| 算子类型 | 推荐块大小 | 网格配置 |
|---------|-----------|---------|
| Element-wise | 256 / 512 | 一维 |
| Reduce | 256 | 一维 + 共享内存 |
| MatMul | dim3(16,16) 或 dim3(32,32) | 二维 |
| 图像处理 | dim3(16,16) | 二维 |

### 1.2 内存访问优化

#### 合并访问 (Coalesced Access)
连续线程访问连续内存地址，GPU 将多次请求合并为少量内存事务。

```cuda
// ✅ 合并访问（连续线程访问连续地址）
__global__ void coalesced(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // 连续访问
    }
}

// ❌ 非合并访问（跳跃访问）
__global__ void strided(float* data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] = data[idx * stride] * 2.0f;  // 跳跃访问
    }
}
```

#### 对齐访问
数据按 128 字节边界对齐，提高内存带宽利用率。

#### 避免 Bank 冲突
共享内存由 32 个 bank 组成，避免同一 warp 内多个线程访问同一 bank。

```cuda
// ✅ 无 bank 冲突
__shared__ float s[256];
s[threadIdx.x] = input[idx];  // 连续线程访问连续 bank

// ❌ bank 冲突
s[threadIdx.x * 32] = input[idx];  // 所有线程访问同一 bank
```

### 1.3 计算优化

#### 避免分支发散
同一 warp 内的 32 个线程应执行相同的控制路径。

```cuda
// ❌ 分支发散：同一 warp 内线程走不同路径
if (threadIdx.x % 2 == 0) {
    // 偶数线程路径
} else {
    // 奇数线程路径
}

// ✅ 使用条件赋值替代分支
float result = (threadIdx.x % 2 == 0) ? value_a : value_b;
```

#### 使用内置快速数学函数
```cuda
// 标准精度
float r = expf(x);

// ✅ 快速版本（精度略低但速度更快）
float r = __expf(x);
float r = __logf(x);
float r = __sinf(x);
```

#### 减少原子操作
尽量使用块内归约代替全局原子操作。

```cuda
// ❌ 大量原子操作
atomicAdd(&global_sum, local_val);

// ✅ 先块内归约，再原子写回
__shared__ float sdata[256];
sdata[tid] = local_val;
__syncthreads();

// 块内归约
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}

// 只有一次原子操作
if (tid == 0) atomicAdd(&global_sum, sdata[0]);
```

### 1.4 Occupancy 优化

- **寄存器使用**: 减少每个线程的寄存器使用量，增加并发 block 数
- **共享内存**: 合理使用共享内存，不超过硬件限制
- **Block 大小**: 选择能整除 SM 最大线程数的 block 大小

## 2. 数值稳定性技巧

### 2.1 防溢出处理

```cuda
// Softmax 数值稳定化：先减去最大值
float max_val = -INFINITY;
for (int i = 0; i < n; i++) {
    max_val = fmaxf(max_val, input[i]);
}

float sum = 0.0f;
for (int i = 0; i < n; i++) {
    sum += __expf(input[i] - max_val);
}

float result = __expf(input[idx] - max_val) / sum;
```

### 2.2 精度提升

- **中间计算**: 使用 `float` 类型即可（无需提升到 `double`）
- **累加操作**: 使用高精度累加器防止精度丢失
- **避免数值下溢**: 检查除零和负数开方操作

```cuda
// 安全除法
float safe_div = (denominator != 0.0f) ? numerator / denominator : 0.0f;

// 安全开方
float safe_sqrt = sqrtf(fmaxf(variance + eps, 0.0f));
```

### 2.3 防负值开方

```cuda
// 方差计算前确保非负
float variance = fmaxf(var_computed, 0.0f);
float std = sqrtf(variance + eps);
```

## 3. 编程约束与最佳实践

### 3.1 必须遵循的规则

- **边界检查**: 所有数组访问前必须检查边界
- **错误检查**: 每个 CUDA API 调用后检查返回值
- **内存对齐**: 确保数据按合适边界对齐
- **线程同步**: 共享内存使用前后必须 `__syncthreads()`

### 3.2 内核设计原则

- **单一职责**: 每个内核只做一件事
- **参数简单**: 避免复杂的数据结构传递
- **内存局部性**: 尽量访问相邻内存位置
- **避免动态分配**: 内核内不使用 `malloc` / `new`

### 3.3 代码规范

- 添加充分的注释说明计算逻辑
- 使用描述性的变量名
- 保持内核函数简洁明了
- 统一的错误处理模式

### 3.4 ⚠️ 禁止事项

- **禁止测试代码**: 生成的内核代码不要包含测试片段
- **禁止打印语句**: 不使用 `printf()`
- **禁止异常抛出**: 不使用 `throw std::runtime_error()` 等
- **禁止动态分配**: 内核内不使用 `malloc` / `new`

## 4. 调试与排查清单

### 内存访问问题
- [ ] 所有数组访问是否都有边界检查？
- [ ] 内存是否正确分配和释放？
- [ ] 主机-设备内存拷贝方向是否正确？
- [ ] 指针是否在有效范围内？

### 内核执行问题
- [ ] 网格和块大小是否合理？
- [ ] 内核启动参数是否正确？
- [ ] 是否有未检查的 CUDA 错误？
- [ ] 设备内存是否足够？

### 性能问题
- [ ] 块大小是否为 2 的幂？
- [ ] 内存访问是否合并？
- [ ] 是否避免了分支发散？
- [ ] 共享内存使用是否高效？
- [ ] 是否有不必要的原子操作？

### 同步问题
- [ ] 共享内存读写前后是否有 `__syncthreads()`？
- [ ] 同步点是否所有线程都能到达（避免死锁）？
- [ ] 是否需要 `__threadfence()` 保证全局可见性？

## 5. 常见错误速查

| 错误类型 | 症状 | 解决方案 |
|---------|------|---------|
| 越界访问 | 运行时错误或结果异常 | 添加边界检查 `if (idx < n)` |
| 内存泄漏 | 程序内存持续增长 | 检查 `cudaFree` 调用 |
| 同步错误 | 结果不确定/不一致 | 添加 `__syncthreads()` |
| 类型不匹配 | 编译错误 | 检查数据类型转换 |
| 设备内存不足 | 内核启动失败 | 减少块大小或分批处理 |
| Bank 冲突 | 性能不佳 | 调整共享内存访问模式 |
| 分支发散 | 性能不佳 | 使用条件赋值替代分支 |
| 非合并访问 | 内存带宽利用率低 | 调整数据布局和访问模式 |

## 最佳实践总结

1. **先正确性后性能**: 确保内核正确性后再优化
2. **合并内存访问**: 连续线程访问连续地址
3. **合理块大小**: 使用 256 或 512，为 2 的幂
4. **避免分支发散**: 同一 warp 内线程走相同路径
5. **使用共享内存**: 缓存频繁访问的数据
6. **减少原子操作**: 先块内归约再全局写回
7. **数值稳定**: 防溢出处理，安全除法和开方
8. **JIT 集成**: 使用 `load_inline` 与 PyTorch 集成
