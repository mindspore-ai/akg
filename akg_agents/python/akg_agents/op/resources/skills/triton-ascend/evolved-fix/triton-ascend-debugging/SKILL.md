---
name: triton-ascend-debugging
description: "Triton Ascend 失败后的调试定位清单和常见错误速查表。仅在验证、编译或运行失败后使用，用于帮助 Conductor 解释报错、定位根因并给出修复建议。生成期硬约束请参考 api-rules、hardware-constraints、memory、grid-config 和对应 guide。"
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A2, Atlas A3, Atlas A5"
---

# Triton Ascend 调试与排查清单

> 本 skill 只用于失败后的诊断和修复建议，不作为初始生成约束注入 KernelGen。
> 如果结论属于“生成时必须遵守”的规则，应落回 fundamental 或对应 guide。

## 完整调试清单

### 内存访问问题
- [ ] tail block 的 `tl.load` / `tl.store` 是否都有 `mask` 或 `boundary_check`？
- [ ] 若 verifier 的错误位置集中在最后一维末尾，优先怀疑 tail mask / boundary_check 漏写。
- [ ] stride 参数是否和 wrapper 中传入的 `tensor.stride()` 一致？
- [ ] 若 kernel 使用一维 `ptr + offsets`，输入是否先 `.contiguous()`？

### 控制流检查
- [ ] 是否误用了 return/break/continue？
- [ ] 复杂条件是否导致 `tl.where` / mask 组合过深？
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
- [ ] 是否对 `tl.sum` / `tl.max` 这类 reduction scalar 结果做了 `[0]` 索引？
- [ ] 切片操作是否使用了正确的API（`tl.get_element`、`tl.extract_slice`等）？

### 性能优化检查
- [ ] 内存访问是否连续（避免跨步访问）？
- [ ] 是否充分利用了块内并行？
- [ ] 复杂算子是否考虑拆分为多个简单kernel？

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

### 验证失败定位

| 错误位置分布 | 优先怀疑 | 修复方向 |
|-------------|---------|---------|
| 只集中在最后一维尾部，例如 `dim2: [16:17]` | tail mask / `boundary_check` 漏写 | 检查最后一维 load/store mask，block_ptr 加 `boundary_check=(0, 1)` |
| 某一整行或整列错误 | 行列 offset、stride、转置或 `sub_vec_id` 偏移错误 | 复核多维 index 分解和 block_ptr `shape/strides/offsets/order` |
| 所有 tile 都像最后一个 tile | A5 亲和单 buffer 被覆盖 | cube 每个 tile fixpipe 后 wait vector 释放，vector 读完后 set buffer-free |
| 误差随 K 增大明显变大 | fp16 长链累加精度损失 | fp32 accumulator；必要时参考 Kahan precision fix |
