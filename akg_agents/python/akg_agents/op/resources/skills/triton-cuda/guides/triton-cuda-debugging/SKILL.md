---
name: triton-cuda-debugging
description: "Triton CUDA 调试排查清单和常见错误速查表，包括编译错误、运行时错误、精度问题和性能问题的诊断方法。适用于 CUDA 内核代码出现错误需要定位原因、或需要验证代码正确性的调试场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: triton_cuda
---

# 调试与排查清单

## 完整调试清单

### 内存访问问题
- [ ] 所有 load/store 是否都有 mask 或 boundary_check？
- [ ] stride 参数设置是否正确？
- [ ] 数组索引是否越界？
- [ ] 是否使用了 `.contiguous()` 确保内存连续？
- [ ] 2D 数据是否使用了 `tl.make_block_ptr`？
- [ ] 内存访问是否合并（coalesced）？

### 控制流检查
- [ ] 是否误用了 return/break/continue？
- [ ] 复杂条件是否用 mask 组合实现？
- [ ] `tl.constexpr` 是否只在内核参数中使用？
- [ ] 是否有 lambda 表达式（不支持）？

### Grid 与 Block 配置检查
- [ ] BLOCK_SIZE 是否为 2 的幂？
- [ ] num_warps 是否合理（2-8）？
- [ ] num_stages 是否合理（2-5）？
- [ ] Grid 总大小是否不超过硬件限制？

### 并发与原子操作检查
- [ ] 并发写入是否使用了原子操作（`tl.atomic_add` 等）？
- [ ] 原子操作是否必要（能否避免）？
- [ ] 是否有数据竞争（多个程序写同一位置）？

### 性能优化检查
- [ ] 是否使用了 autotune？
- [ ] MatMul 是否使用了 Grouped Ordering？
- [ ] 是否使用 float32 进行中间累加？
- [ ] Reduce 操作是否有数值稳定性处理？

## 常见错误速查表

### 编译错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **Return 语句** | 编译失败 | Kernel 中使用 return | 移除 return，使用 mask 代替 |
| **Break/Continue** | 编译失败 | 不支持控制流跳转 | 用 mask 或重构逻辑 |
| **Lambda 表达式** | 编译失败 | 不支持 lambda | 改用普通函数或内联 |
| **类型错误** | 编译失败 | constexpr 类型不匹配 | 检查 tl.constexpr 声明 |

### 运行时错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **内存越界** | CUDA error | 缺少 mask | 添加 mask 或 boundary_check |
| **形状不匹配** | 维度错误 | stride 计算错误 | 检查 stride 参数 |
| **非法内存访问** | Segfault | 指针计算错误 | 验证偏移计算 |
| **共享内存溢出** | Launch failed | num_stages 过大 | 减少 num_stages |

### 数值错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **NaN/Inf** | 结果异常 | Softmax 溢出 | 减去最大值 |
| **精度损失** | 结果不准确 | 全程使用 fp16 累加 | 使用 float32 累加 |
| **除零错误** | NaN | 方差或和为零 | 添加 eps |
| **负数开方** | NaN | 方差为负 | `tl.maximum(var, 0.0)` |

### 性能问题

| 问题类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **性能差** | 比 PyTorch 慢 | 未使用 autotune | 添加 autotune |
| **带宽低** | 内存受限 | 非合并访问 | 确保合并访问 |
| **Occupancy 低** | GPU 利用率低 | 寄存器/共享内存超限 | 减小 BLOCK_SIZE |
| **L2 缓存差** | MatMul 性能低 | 未使用 Grouped Ordering | 添加 L2 缓存优化 |

## 分类调试流程

### 1. 编译失败

**步骤**:
1. 检查错误信息中的关键词（return, break, lambda）
2. 查看是否使用了不支持的语法
3. 参考"API 使用限制"部分修改代码

**常见修复**:
```python
# 错误：使用 return
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= n:
        return  # 编译错误！
    # ...

# 正确：使用 mask
@triton.jit
def kernel(ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    data = tl.load(ptr + offsets, mask=mask, other=0.0)
    # ... 所有代码在同一层级
```

### 2. 运行时崩溃

**步骤**:
1. 添加所有 load/store 的 mask
2. 检查 Grid 和 BLOCK_SIZE 配置
3. 验证 stride 参数是否正确
4. 使用小数据测试
5. 检查共享内存是否超限（减少 num_stages）

**调试技巧**:
```python
# 打印调试信息（host 侧）
print(f"Grid: {grid}, BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Shape: {input_tensor.shape}, Stride: {input_tensor.stride()}")
print(f"Contiguous: {input_tensor.is_contiguous()}")
```

### 3. 结果不正确

**步骤**:
1. 检查数值稳定性（Softmax 是否减去最大值）
2. 验证累加精度（是否使用 float32）
3. 检查边界处理（mask 是否正确）
4. 对比小规模手算结果

**验证方法**:
```python
# 与 PyTorch 原生实现对比
output_triton = model_new(x)
output_torch = torch.softmax(x, dim=-1)  # 或其他原生实现
diff = (output_triton - output_torch).abs().max()
print(f"Max diff: {diff.item()}")
assert diff < 1e-5, "Results mismatch!"
```

### 4. 性能不佳

**步骤**:
1. 添加 autotune 搜索最优配置
2. 检查是否转为连续内存（`.contiguous()`）
3. 确认内存访问是否合并
4. 检查 L2 缓存优化（Grouped Ordering）
5. 使用 Nsight Compute 分析

**性能分析**:
```python
import time

# 预热
for _ in range(10):
    _ = model(x)

# 测试
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    _ = model(x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Average time: {elapsed/100*1000:.2f} ms")
```

## 错误修复示例

### 示例 1: Softmax 溢出

**错误代码**:
```python
numerator = tl.exp(x)  # 可能溢出
```

**修复**:
```python
max_val = tl.max(x, axis=0)
x_stable = x - max_val
numerator = tl.exp(x_stable)
```

### 示例 2: 非合并访问

**错误代码**:
```python
# 每个线程跳跃访问
offsets = pid + tl.arange(0, BLOCK_SIZE) * stride
data = tl.load(ptr + offsets)
```

**修复**:
```python
# 连续访问
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
data = tl.load(ptr + offsets, mask=offsets < n)
```

### 示例 3: 共享内存溢出

**错误代码**:
```python
triton.Config({...}, num_stages=8, num_warps=8)  # 共享内存不足
```

**修复**:
```python
triton.Config({...}, num_stages=3, num_warps=4)  # 减少 stage 数
```

## 调试工具

### 1. Nsight Compute

```bash
# 分析 kernel 性能
ncu --set full python script.py

# 分析内存带宽
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python script.py
```

### 2. CUDA-MEMCHECK

```bash
# 检查内存错误
compute-sanitizer python script.py
```

### 3. 使用小数据测试

```python
# 大数据难以调试，先用小数据
x_small = torch.randn(4, 8, device='cuda', dtype=torch.float16)
output = model(x_small)
print(output)  # 手动验证结果
```

### 4. 对比参考实现

```python
# 始终与 PyTorch 原生实现对比
torch.testing.assert_close(output_triton, output_torch, rtol=1e-4, atol=1e-5)
```

## 总结

调试 Triton-CUDA 代码的关键：
1. **遵守规范**: 不使用 return/break/continue/lambda
2. **内存安全**: 所有访问都加 mask
3. **数值稳定**: Softmax 减最大值，float32 累加
4. **合并访问**: 确保同一 warp 内线程访问连续地址
5. **性能优化**: 使用 autotune、Grouped Ordering

**最佳实践**: 先保证正确性，再优化性能！
