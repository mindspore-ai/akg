# Triton 专家技巧与优化建议

本文档提供 Triton 开发的技巧、性能优化和问题排查指南。

## 1. 特定算子优化

### matmul 算子

**注意**：合理的切分是提升matmul算子性能的关键。

#### Ascend后端切分优化：充分发挥带宽，算子行宽为512B的整数倍，且单次行数尽量大，以fp16和bf16为例：
- **A、B都不转置**：分块行宽分别为K0和N0，则M0=128，K0=256，N0=256
- **A不转置，B转置**：分块行宽都是K0，则K0=256，M0和N0影响较小
- **A、B都转置**：分块行宽分别为M0和K0，则M0=256，K0=256，N0=128
- **A转置，B不转置**：分块行宽分别为M0和N0，则左右矩阵均无法同时满足512B的整数倍，需根据实际情况调整

### Attention 算子

#### 标准Attention计算流程：
1. **QK^T计算**：`scores = Q @ K^T / sqrt(d_k)`，计算注意力分数
2. **Softmax归一化**：`attn_weights = softmax(scores)`，确保权重和为1
3. **加权求和**：`output = attn_weights @ V`，得到最终输出

#### Flash Attention优化策略：
- **分块计算**：将大矩阵分块处理，减少内存占用
- **在线Softmax**：使用增量式softmax算法，分块计算，维护全局最大值和归一化因子，避免存储完整注意力矩阵，具体逻辑如下：
```python
# 初始化全局统计量
m_i = -float("inf")  # 全局最大值
l_i = 0.0           # 全局exp和
acc = 0.0           # 输出累加器

# 分块处理
for start_n in range(0, seq_len, BLOCK_SIZE):
    # 1. 加载当前块的分数
    scores = tl.load(scores_ptr + start_n, mask=load_mask, other=-float("inf"))
    
    # 2. 更新全局最大值
    m_ij = tl.maximum(m_i, tl.max(scores, 0))
    
    # 3. 计算当前块的exp值（数值稳定化）
    scores = scores - m_ij
    p = tl.math.exp2(scores)
    
    # 4. 更新全局exp和
    l_ij = tl.sum(p, 0)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    
    # 5. 更新输出累加器
    acc = acc * alpha + p
    
    # 6. 更新全局最大值
    m_i = m_ij

# 最终归一化
acc = acc / l_i
```

## 2. 性能优化

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

## 3. 数值稳定性技巧

### 防溢出处理
```python
# 归一化前先减去最大值
max_val = tl.max(data, axis=0)
stable_data = data - max_val
exp_data = tl.exp(stable_data)
```

### 防负值开方处理
- 在任何平方根操作前，确保被开方数是非负的（例如，使用 max(input, 0.)或 max(input, eps)）

### 精度提升
- **中间计算**: 关键步骤转为float32提升精度
- **累加操作**: 使用高精度累加器防止精度丢失

## 4. API使用限制与替代方案

### 禁止使用的语法
- 禁止 `return`, `break`, `continue` → 使用mask控制
- 禁止 lambda表达式 → 使用内联函数或tl.where
- 禁止 链式布尔运算 → 分步计算mask
- 禁止 张量直接索引 → 使用tl.load/tl.store
**Ascend后端**
- 禁止 `tl.where` → 使用if-else

### tl.constexpr 正确用法
- **仅在内核参数中使用**: `BLOCK_SIZE: tl.constexpr`
- **不可在host侧使用**: 启动函数中不可用tl.constexpr

### 输出张量创建规范
- host 侧使用 `torch.empty` 或 `torch.empty_like` 创建输出张量
- 不要使用 `torch.zeros` 或 `torch.ones`，避免不必要的初始化开销

### Ascend 后端避免使用 tl.where 计算内存偏移
Ascend 后端对`tl.where`生成的复杂指针运算支持不完全。复杂条件判断可以采用if-else静态分支处理，而非在内存访问时动态计算。

### 标量类型转换
- **仅支持to(type)**: 如`scalar.to(tl.float16)`，禁止使用`tl.float16(scalar)`
- **tl.constexpr类型转换**: 将常量赋值给临时变量再转换，如`scalar = CONST_A`

### 切分设置
**Ascend后端**
- BLOCK_SIZE必须小于65536，并且线程块所占内存必须符合硬件限制
- 若shape过大，单次切分后超过硬件缓存，并且BLOCK_SIZE超过限制，可以对循环进行多次切分

## 5. 调试与排查清单

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

### 切分设置
内核启动网格大小必须不超过65535，且线程块所占的内存必须符合硬件限制。

如果当前切分超出硬件缓存，且内核启动网格大小会超过限制，可以通过以下方法改进：
**对张量进行多次切分：**如果一次内核启动无法处理整个张量，可以将张量分成多个部分，通过多次启动内核来完成计算
```python
if M > max_grid_size:
    # 使用多批次处理大网络：在循环中多次启动内核
    for start_row in range(0, M, max_grid_size):
        end_row = min(start_row + max_grid_size, M)
        batch_size = end_row - start_row

        # 提取当前批次的数据
        x_batch = x[start_row:end_row]

        # 启动1D内核处理当前批次
        grid = (batch_size, )
        op_kernel[grid](x_batch, x_batch.stride(0), BLOCK_SIZE)
```

## 5. 常见错误速查

| 错误类型 | 症状 | 解决方案 |
|---------|------|---------|
| 越界访问 | 运行时错误或结果异常 | 添加mask或boundary_check |
| 控制流错误 | 编译失败 | 移除return/break，使用mask |
| 类型不匹配 | 编译警告或错误 | 检查tl.constexpr类型 |
| 数据错位 | 计算结果错误 | 验证stride设置 |
| 竞争条件 | 结果不确定 | 使用原子操作 |

## 7. 开发建议

### 代码风格
- 添加充分的注释说明计算逻辑
- 使用描述性的变量名
- 保持内核函数简洁明了
