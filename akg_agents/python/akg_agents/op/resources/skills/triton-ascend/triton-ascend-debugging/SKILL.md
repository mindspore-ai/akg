---
name: triton-ascend-debugging
description: "调试排查清单和常见错误速查表"
level: L4
category: implementation
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton-ascend
---

# 调试与排查清单

## 完整调试清单

### 内存访问问题
- [ ] 所有 load/store 是否都有 mask 或 boundary_check？
- [ ] stride 参数设置是否正确？
- [ ] 数组索引是否越界？
- [ ] 是否使用了 `.contiguous()` 确保内存连续？
- [ ] 2D 数据是否使用了 `tl.make_block_ptr`？

### 控制流检查
- [ ] 是否误用了 return/break/continue？
- [ ] 是否使用了 while 循环（Ascend 不支持）？
- [ ] 复杂条件是否用 mask 组合实现？
- [ ] `tl.constexpr` 是否只在内核参数中使用？
- [ ] 是否有 lambda 表达式（不支持）？

### Grid 与 Block 配置检查
- [ ] Grid 总大小是否不超过 65535？
- [ ] 对于大 shape 算子，是否采用了交错循环 `for i in range(pid, total, core_num)`？
- [ ] Grid 维度是否为 tuple 类型且不超过 3 维？
- [ ] BLOCK_SIZE 是否小于 65536？
- [ ] 核心数是否在 `__init__` 中获取（而非 forward）？

### 并发与原子操作检查
- [ ] 并发写入是否使用了原子操作（`tl.atomic_add` 等）？
- [ ] 原子操作是否必要（能否避免）？
- [ ] 是否有数据竞争（多个程序写同一位置）？

### 切片与索引检查
- [ ] 是否使用了不支持的切片语法（`arr[i:j]`）？
- [ ] 是否使用了 `tl.get_element` 或 `tl.extract_slice`？
- [ ] 是否对 `tl.arange` 使用了 `get_element`（禁止）？

### 性能优化检查
- [ ] BLOCK_SIZE 是否对齐到 256B（fp16: 128, fp32: 64）？
- [ ] 是否使用了 VEC/CUBE 核心数？
- [ ] 是否使用 float32 进行中间累加？
- [ ] Reduce 操作是否有数值稳定性处理？

## 常见错误速查表

### 编译错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **While 循环** | 编译失败 | Ascend 不支持 while | 改用 `for + if` 替代 |
| **Return 语句** | 编译失败 | Kernel 中使用 return | 移除 return，使用 store 代替 |
| **Break/Continue** | 编译失败 | 不支持控制流跳转 | 用 mask 或重构逻辑 |
| **Lambda 表达式** | 编译失败 | 不支持 lambda | 改用普通函数或内联 |
| **切片语法** | 编译失败 | 使用了 `arr[i:j]` | 使用 `tl.extract_slice` |

### 运行时错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| **内存越界** | 运行时崩溃 | 缺少 mask | 添加 mask 或 boundary_check |
| **形状不匹配** | 维度错误 | stride 计算错误 | 检查 stride 参数 |
| **Grid 超限** | 启动失败 | grid > 65535 | 使用交错循环处理 |
| **设备同步** | 性能下降 | forward 中调用 `get_device_limit` | 移到 `__init__` |

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
| **性能差** | 比预期慢 | 未使用 block_ptr | 使用 `tl.make_block_ptr` |
| **启动开销大** | Grid 数量过多 | 未使用固定核心数 | 交错循环 + 固定 grid |
| **内存访问慢** | 带宽未饱和 | 非连续访问 | 调用 `.contiguous()` |
| **缓存命中低** | 性能不佳 | 随机访问模式 | 优化数据布局 |

## 分类调试流程

### 1. 编译失败

**步骤**:
1. 检查错误信息中的关键词（while, return, break, lambda）
2. 查看是否使用了不支持的语法
3. 参考"API 使用限制"部分修改代码

**常见修复**:
```python
# ❌ 错误：while 循环
while condition:
    do_something()

# ✅ 正确：for + if
for i in range(MAX_ITERS):
    if i < condition:
        do_something()
```

### 2. 运行时崩溃

**步骤**:
1. 添加所有 load/store 的 mask
2. 检查 Grid 和 BLOCK_SIZE 配置
3. 验证 stride 参数是否正确
4. 使用小数据测试

**调试技巧**:
```python
# 添加 assert 检查
assert grid[0] < 65536, f"Grid size {grid[0]} exceeds limit"
assert BLOCK_SIZE < 65536, f"Block size {BLOCK_SIZE} exceeds limit"

# 打印调试信息（host 侧）
print(f"Grid: {grid}, BLOCK_SIZE: {BLOCK_SIZE}")
print(f"Shape: {input_tensor.shape}, Stride: {input_tensor.stride()}")
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
1. 检查是否转为连续内存（`.contiguous()`）
2. 验证 BLOCK_SIZE 是否对齐（256B）
3. 确认使用了正确的核心数（VEC/CUBE）
4. 检查 Grid 配置是否合理
5. 使用 Profiler 分析瓶颈

**性能分析**:
```python
import time

# 预热
for _ in range(10):
    _ = model(x)

# 测试
torch.cuda.synchronize()  # 或 torch.npu.synchronize()
start = time.time()
for _ in range(100):
    _ = model(x)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Average time: {elapsed/100*1000:.2f} ms")
```

## 错误修复示例

### 示例 1: Grid 超限

**错误代码**:
```python
n_elements = 10000000
BLOCK_SIZE = 1024
grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # 约 9766 块，可能超限
```

**修复**:
```python
# 使用交错循环
grid = (self.VEC_CORE_NUM,)
# 在 kernel 中: for i in range(pid, total_blocks, CORE_NUM)
```

### 示例 2: Softmax 溢出

**错误代码**:
```python
numerator = tl.math.exp2(x * 1.44269504)  # 可能溢出
```

**修复**:
```python
max_val = tl.max(x, axis=0)
x_stable = x - max_val
numerator = tl.math.exp2(x_stable * 1.44269504)
```

### 示例 3: While 循环

**错误代码**:
```python
i = 0
while i < n_iters:
    process(i)
    i += 1
```

**修复**:
```python
for i in range(MAX_ITERS):
    if i < n_iters:
        process(i)
```

## 调试工具和技巧

### 1. 使用 assert 验证

```python
# Host 侧
assert x.is_contiguous(), "Input must be contiguous"
assert grid[0] < 65536, f"Grid size exceeds limit"

# Kernel 内（通过 mask 实现）
valid = offsets < n_elements
data = tl.load(ptr + offsets, mask=valid, other=0.0)
```

### 2. 分步骤调试

从简单开始，逐步增加复杂度：
1. 先实现最简单的版本（可能性能差）
2. 验证结果正确性
3. 逐步添加优化
4. 每次优化后验证正确性

### 3. 对比参考实现

始终与 PyTorch/MindSpore 原生实现对比：
```python
# 测试正确性
torch.testing.assert_close(output_triton, output_torch, rtol=1e-4, atol=1e-5)
```

### 4. 使用小数据测试

大数据难以调试，先用小数据：
```python
# 测试
x_small = torch.randn(4, 8, device='npu', dtype=torch.float16)
output = model(x_small)
print(output)  # 手动验证结果
```

## 快速检查命令

```bash
# 检查代码中的常见错误
grep -n "while " kernel.py  # 查找 while 循环
grep -n "return " kernel.py  # 查找 return 语句
grep -n "break\|continue" kernel.py  # 查找 break/continue
grep -n "lambda" kernel.py  # 查找 lambda

# 检查 Grid 配置
grep -n "grid = " kernel.py | grep -v "tuple"  # 查找非 tuple 的 grid
```

## 总结

调试 Triton-Ascend 代码的关键：
1. **遵守规范**: 不使用 while/return/break/continue/lambda
2. **注意限制**: Grid < 65535, BLOCK_SIZE < 65536
3. **数值稳定**: Softmax 减最大值，float32 累加
4. **内存安全**: 所有访问都加 mask
5. **性能优化**: 连续内存、对齐、固定核心数

**最佳实践**: 先保证正确性，再优化性能！
