---
name: triton-cuda-optimization
description: "Triton CUDA 性能优化通用策略、API 限制说明和调试技巧汇总。适用于需要提升 GPU 内核性能、遇到编译/运行错误需要排查、或需要了解 CUDA 平台限制的内核代码生成和优化场景"
category: method
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
structure:
  child_skills:
    - triton-cuda-memory
    - triton-cuda-grid-config
    - triton-cuda-debugging
---

# Triton CUDA 性能优化指南

## 1. 性能优化策略

### 1.1 块大小选择

- **原则**: 平衡并行度与资源占用
- **建议**: 使用 2 的幂次（256, 512, 1024）
- **GPU 考量**: 需要足够多的 warp 来隐藏延迟

### 1.2 Warp 和 Stage 调优

CUDA 后端特有的两个重要参数：

- **num_warps**: 每个 block 的 warp 数量（每个 warp = 32 个线程）
  - 小 BLOCK_SIZE：使用较少的 warp (2-4)
  - 大 BLOCK_SIZE：使用较多的 warp (4-8)
  - MatMul：通常使用 4-8 个 warp

- **num_stages**: 软件流水线级数
  - 更多的 stage 可以更好地隐藏内存延迟
  - 但会占用更多共享内存
  - 通常 2-5 之间选择

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=4),
    ],
    key=['n_elements'],
    restore_value=['output_ptr'],  # 必须：列出所有输出指针参数名
)
```

### 1.3 内存访问优化

- **合并访问 (Coalesced Access)**: 同一 warp 内的线程应访问连续内存地址
- **2D数据**: 优先使用 `tl.make_block_ptr` 配合 `boundary_check`
- **步幅设计**: 仔细设计 stride 参数，错误设置会严重影响性能
- **数据布局**: 保持内存访问的连续性和局部性

### 1.4 算子拆分策略

- **复杂算子**: 拆分为多个简单 kernel，避免单个 kernel 过于复杂
- **融合策略**: 适度融合以减少全局内存读写（如 fused attention）
- **平衡**: CUDA 后端融合通常比 NPU 更有效，但仍需注意 register pressure

### 1.5 Occupancy 优化

GPU 利用率（Occupancy）是性能的关键指标：

- **寄存器使用**: 减少每个线程的寄存器使用量，增加并发 block 数
- **共享内存**: 合理使用共享内存，不超过硬件限制
- **Block 大小**: 选择能整除 SM 最大线程数的 block 大小

## 2. 数值稳定性

### 2.1 防溢出处理

**Softmax 数值稳定化**:
```python
# 减去最大值防止 exp 溢出
max_val = tl.max(scores, axis=0)
scores = scores - max_val
p = tl.exp(scores)  # CUDA 后端直接使用 tl.exp
```

### 2.2 防负值开方

```python
# 方差计算前确保非负
variance = tl.maximum(variance, 0.0)
std = tl.sqrt(variance + eps)
```

### 2.3 精度提升

- **使用 float32 进行累加**: 即使输入是 float16/bfloat16
- **最后再转换**: 计算完成后再转回目标精度
- **TF32**: Ampere+ GPU 上可使用 TF32 加速 MatMul

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
# ... 累加计算 ...
result = tl.cast(accumulator, output_dtype)
```

## 3. API 使用限制

### 3.1 禁止使用的语法

**禁止使用**: `return`, `break`, `continue`, `lambda`

Triton 内核是一次性执行完整逻辑，不支持提前返回或跳转语句。

```python
# 错误：使用 return
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return  # 编译错误！

# 正确：使用 mask
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    # ... 所有代码都在同一层级执行
```

### 3.2 tl.constexpr 正确用法

- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在 host 侧使用**: 启动函数中不可用 tl.constexpr

### 3.3 输出张量创建规范

- 正确：使用 `torch.empty` 或 `torch.empty_like`
- 错误：避免 `torch.zeros` 或 `torch.ones`（避免不必要的初始化开销）

### 3.4 Conv 类卷积算子编写注意

torch module 中的卷积算子生成会包含一个随机权重 weight，为保证 triton 实现的结果一致，需要在 host 侧代码中生成对应的 weight：

```python
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_kernel():
    pass

def triton_host():
    args = ...
    weight = nn.Conv2d(**args).weight.to(device)
```

具体的参数和 `nn` 中调用的 module 要与 torch 保持一致，device 设置为 `"cuda"`。调用 triton 之前会固定相同的随机种子，只需正确创建类的实例并导出权重。

## 4. 性能检查清单

### 内存访问
- [ ] 内存访问是否合并（coalesced）？
- [ ] 是否使用了 2D block_ptr 优化多维数据访问？
- [ ] 是否保证了内存访问的连续性？

### 并行度配置
- [ ] BLOCK_SIZE 是否为 2 的幂？
- [ ] num_warps 是否合理（2-8）？
- [ ] num_stages 是否合理（2-5）？

### 算子设计
- [ ] 复杂算子是否需要拆分？
- [ ] 是否合理使用了算子融合？
- [ ] 是否使用了 autotune？

### 数值稳定性
- [ ] Reduce 操作是否有防溢出处理？
- [ ] 是否使用 float32 进行中间累加？
- [ ] 是否处理了除零、负数开方等边界情况？

## 最佳实践总结

1. **Autotune**: 使用 autotune 搜索最优 BLOCK_SIZE、num_warps、num_stages
2. **内存合并**: 确保同一 warp 内线程访问连续地址
3. **Tensor Core**: MatMul 类算子启用 allow_tf32
4. **流水线**: 通过 num_stages 隐藏内存延迟
5. **数值稳定**: 使用 float32 累加，减去最大值防溢出
6. **Occupancy**: 平衡寄存器和共享内存使用
