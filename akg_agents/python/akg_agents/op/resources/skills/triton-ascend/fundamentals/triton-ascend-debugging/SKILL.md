---
name: triton-ascend-debugging
description: "Triton Ascend 调试排查清单和常见错误速查表，包括编译错误、运行时错误、精度问题和性能问题的诊断方法。适用于内核代码生成、出现错误需要定位原因、或需要验证代码正确性的调试场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3"
---

# 调试与排查清单

## 完整调试清单

### 内存访问问题
- [ ] 所有 load/store 是否都有 mask 或 boundary_check？
- [ ] stride 参数设置是否正确？
- [ ] 数组索引是否越界？

### 控制流检查
- [ ] 是否误用了 return/break/continue？
- [ ] 复杂条件是否用 mask 组合实现？
- [ ] `tl.constexpr` 是否只在内核参数中使用？

### Grid 与 Block 配置检查
- [ ] Grid 总大小是否不超过 65535？
- [ ] 对于大 shape 算子，是否采用了交错循环 `for i in range(pid, total, core_num)`？
- [ ] Grid 维度是否为 tuple 类型且不超过 3 维？

### 并发与原子操作检查
- [ ] 并发写入是否使用了原子操作（`tl.atomic_add` 等）？
- [ ] 原子操作是否必要（能否避免）？

### 切片与索引检查
- [ ] 是否避免了Python风格的直接切片（如`b[0]`、`b[i:j]`）？
- [ ] 是否对`tl.arange`生成的张量误用了`tl.get_element`？
- [ ] 切片操作是否使用了正确的API（`tl.get_element`、`tl.extract_slice`等）？

### 性能优化检查
- [ ] 内存访问是否连续（避免跨步访问）？
- [ ] 是否充分利用了块内并行？
- [ ] 复杂算子是否考虑拆分为多个简单kernel？

## 禁止使用的语法（Ascend 后端）

| 禁止写法 | 替代方案 |
|---------|---------|
| `return` / `break` / `continue` | 使用 mask 控制流程 |
| lambda 表达式 | 内联函数或 tl.where |
| 链式布尔运算 `a and b` | 分步计算 mask：`m1 = ...; m2 = ...; m = m1 & m2` |
| 张量直接索引 `tensor[i]` | `tl.load(ptr + offset)` / `tl.store(ptr + offset, val)` |
| Python 切片 `b[0]` / `b[i:j]` | `tl.get_element` / `tl.extract_slice` / `tl.insert_slice` |
| 对 `tl.arange` 结果用 `get_element` | 直接计算索引值 |
| `while` 循环 | `for i in range(MAX): if i < n:` |
| `range()` 混用运行时变量和 constexpr | 全 constexpr 的 `range(0, N, BLOCK_K)` + 循环体内运行时 if |
| `tl.float16(scalar)` | `scalar.to(tl.float16)` |
| `tl.constexpr` 在 host 侧使用 | 仅在 kernel 参数中使用 |
| if-else 中负偏移 | `tl.maximum(offset, 0)` |
| 复杂 `tl.where` 用于内存偏移 | 拆分为 if-else 静态分支 |

## 常见错误速查表

### 编译错误

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| UB/CBUF 溢出 | `ub overflow, requires X bits while 1572864 bits available` | BLOCK 尺寸过大或中间变量过多 | 缩小 BLOCK 尺寸；减少同时活跃的 tensor 数 |
| HiVM vsel 错误 | `hivm.hir.vsel: Unsupported op for finding the root alloc` | 嵌套 mask + tl.where 组合过于复杂 | 用乘法替代 tl.where：`a * mask.to(dtype)` |
| 内存越界访问 | 运行时错误、结果异常、随机崩溃 | load/store缺少mask或boundary_check | 添加正确的mask或boundary_check保护 |
| Grid超限 | 编译失败或运行时错误 | grid总大小超过65535 | 使用交错循环`for i in range(pid, total, core_num)`或连续分块处理 |
| 控制流错误 | `unsupported AST node type: Continue` | 使用了return/break/continue | 改用 if-else 包裹逻辑 |
| while循环错误 | 编译失败（Ascend后端） | 使用了while循环 | 改用for + if替代：`for i in range(MAX): if i < n:` |
| constexpr 索引 | `ValueError('unsupported tensor index: constexpr[0]')` | 对 tl.sum 等返回的标量做 `[0]` 索引 | 直接使用标量结果，不要索引 |
| 切片语法错误 | 编译失败 | 使用了`b[0]`或`b[i:j]`直接切片 | 使用`tl.get_element`或`tl.extract_slice` |
| tl.arange索引错误 | 编译失败 | 对`tl.arange`结果使用`get_element` | 直接计算索引值而非提取 |
| 类型转换错误 | `cast incompatible` | 隐式 cast 或使用`tl.float16(scalar)` | 用 `.to(tl.float16)` 或显式指定 acc dtype |
| constexpr误用 | 编译失败 | 在host侧使用tl.constexpr | 仅在kernel参数中使用tl.constexpr |
| Stride设置错误 | 计算结果错误、数据错位 | stride参数计算或传递错误 | 验证stride设置，检查tensor.stride() |
| 数值不稳定 | 结果为NaN或Inf | softmax/sqrt等操作溢出 | 减去最大值、检查非负、使用float32 |
| 数据竞争 | 结果不确定、每次运行不同 | 多program并发写入同一位置 | 使用tl.atomic_add等原子操作 |
| BLOCK_SIZE过大 | 编译失败或运行时错误 | BLOCK_SIZE超过65536或硬件限制 | 减小BLOCK_SIZE，使用循环处理 |
| tl.where偏移计算 | 编译失败（Ascend后端） | 在内存偏移中使用tl.where | 改用if-else静态分支处理 |
| 性能低下 | 运行缓慢 | 内存访问不连续、切分不合理 | 优化内存布局、调整BLOCK_SIZE、使用block_ptr |
| 运行时range边界崩溃 | bishengIR crash | range()的start/stop混用运行时变量和constexpr | 改用全constexpr的range(0, N, BLOCK_K)，循环体内用运行时if跳过 |