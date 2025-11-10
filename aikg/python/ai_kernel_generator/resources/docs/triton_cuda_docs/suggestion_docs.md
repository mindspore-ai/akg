# Triton 专家技巧与优化建议

本文档提供 Triton 开发的技巧、性能优化和问题排查指南。

## 1. 性能优化

### 块大小选择策略
- **基础**: 2的幂（256, 512, 1024）
- **Ascend后端**: 可考虑16的倍数
- **调优**: 平衡并行度与资源占用，避免过大或过小

### 内存访问优化
- **2D数据**: 优先使用 `tl.make_block_ptr` 配合 `boundary_check`，自动优化内存合并
- **步幅设计**: 仔细设计stride参数，错误设置会严重影响性能
- **数据布局**: 保持内存访问的连续性和局部性

### 算子拆分策略
- **复杂算子**: 拆分为多个简单kernel，避免单个kernel过于复杂

## 2. 数值稳定性技巧

### 防溢出处理
```python
# 归一化前先减去最大值
max_val = tl.max(data, axis=0)
stable_data = data - max_val
exp_data = tl.exp(stable_data)
```

### 精度提升
- **中间计算**: 关键步骤转为float32提升精度
- **累加操作**: 使用高精度累加器防止精度丢失

## 3. API使用限制与替代方案

### 禁止使用的语法
- 禁止 `return`, `break`, `continue` → 使用mask控制
- 禁止 lambda表达式 → 使用内联函数或tl.where
- 禁止 链式布尔运算 → 分步计算mask
- 禁止 张量直接索引 → 使用tl.load/tl.store

### tl.constexpr 正确用法
- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在host侧使用**: 启动函数中不可用tl.constexpr

## 4. 调试与排查清单

### 内存访问问题
- [ ] 所有load/store是否都有mask或boundary_check？
- [ ] stride参数设置是否正确？
- [ ] 数组索引是否越界？

### 控制流问题  
- [ ] 是否误用了return/break/continue？
- [ ] 复杂条件是否用mask组合实现？
- [ ] tl.constexpr是否只在内核参数中使用？

### 原子操作问题
- [ ] 并发写入是否使用了原子操作？
- [ ] 原子操作的数据类型是否匹配？

### 性能问题
- [ ] BLOCK_SIZE是否为2的幂？
- [ ] 内存访问是否连续？
- [ ] 网格大小是否合理？

## 5. 常见错误速查

| 错误类型 | 症状 | 解决方案 |
|---------|------|---------|
| 越界访问 | 运行时错误或结果异常 | 添加mask或boundary_check |
| 控制流错误 | 编译失败 | 移除return/break，使用mask |
| 类型不匹配 | 编译警告或错误 | 检查tl.constexpr类型 |
| 数据错位 | 计算结果错误 | 验证stride设置 |
| 竞争条件 | 结果不确定 | 使用原子操作 |

## 6. 开发建议

### 代码风格
- 添加充分的注释说明计算逻辑
- 使用描述性的变量名
- 保持内核函数简洁明了

---

# 注意conv类卷积算子编写：
torch module中的卷积算子生成会包含一个随机权重weight，为保证我们生成的triton实现也具有相同的结果，我们可以在triton的host侧代码中生成对应的weight，例如：
'''
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def triton_kernel():
    pass
def triton_host():
    args = ...
    weight = nn.conv(**args).weight.to(device)
'''
具体的参数和''nn''中调用的module要与torch保持一致，device与传入的backend保持一致（如"cuda", "npu")
我会在调用triton之前固定相同的随机种子，所以在这里只需要正确地创建类的实例，并导出权重