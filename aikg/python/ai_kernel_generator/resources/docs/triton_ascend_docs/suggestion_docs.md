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
- **调优**: 平衡并行度与资源占用，避免过大或过小

### 内存访问优化
- **2D数据**: 优先使用 `tl.make_block_ptr` 配合 `boundary_check`，自动优化内存合并
- **步幅设计**: 仔细设计stride参数，错误设置会严重影响性能
- **数据布局**: 保持内存访问的连续性和局部性

#### 连续内存的一维访问优化
张量在内存中连续存储时，可用一维指针遍历，避免多维索引开销：

```python
# 方案1：转连续后用一维访问（推荐）
def launch_kernel(input_tensor):
    # 非连续张量转为连续（一次性开销）
    if not input_tensor.is_contiguous():
        input_tensor = input_tensor.contiguous()
    
    output_tensor = torch.empty_like(input_tensor)
    n_elements = input_tensor.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    elementwise_kernel[grid](input_tensor, output_tensor, n_elements, BLOCK_SIZE)
    return output_tensor

@triton.jit
def elementwise_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    data = tl.load(input_ptr + offsets, mask=mask)
    result = compute(data)
    tl.store(output_ptr + offsets, result, mask=mask)

# 方案2：使用stride访问（不推荐）
# 每次load/store都需计算stride偏移，性能较差
@triton.jit
def elementwise_kernel_stride(input_ptr, output_ptr, M, N, stride_m, stride_n, 
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # 每个元素访问都有额外开销
    ...
```

**性能分析**：
- **`.contiguous()`转换**：一次性内存拷贝开销，后续访问高效（连续内存，缓存友好）
- **stride访问**：每次load/store都需计算偏移，累积开销大，内存访问不连续
- **建议**：非连续张量先调用`.contiguous()`转换，再用一维访问，整体性能更优

**要点**：
- 优先使用`.contiguous()`转换 + 一维访问
- 连续内存访问效率远高于stride计算开销
- `torch.empty_like()`创建的输出默认连续
- 输入输出同形状的element-wise算子无需reshape

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

### 切片操作规范
Triton不支持Python风格的直接切片语法（如`b[0]`或`b[i:j]`），需使用专用API：
- **单元素提取**：`tl.get_element(tensor, (index,))` 或单次load
- **切片提取**：`tl.extract_slice(tensor, offsets, sizes, strides)`
- **切片插入**：`tl.insert_slice(ful, sub, offsets, sizes, strides)` - 将sub张量插入到ful张量的指定位置

```python
# 一维切片插入
output_sub = x_sub + y_sub
output = tl.insert_slice(output, output_sub, [offset], [size], [1])

# 二维切片插入（逐行构建）
tmp_buf = tl.zeros((rows, cols), dtype)
val = tl.load(in_ptr + offset, mask)
tmp_buf = tl.insert_slice(tmp_buf, val[None,:], offsets=(i, 0), sizes=(1, cols), strides=(1, 1))
```

**重要限制**：禁止对`tl.arange`生成的张量使用`get_element()`
- `tl.arange`是编译时索引表达式，非实际张量，需直接计算而非提取
```python
# 错误：offsets = base + tl.arange(0, BLOCK_SIZE); value = tl.get_element(offsets, [i])
# 正确：value = base + i

# 正确用法
element = tl.get_element(tensor, (i, j))  # 实际张量
sub_tensor = tl.extract_slice(tensor, [0], [32], [1])  # 提取切片
```

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

### Grid设置规范
- **维度限制**：grid必须是tuple类型，最多3维，如`(x,)`、`(x, y)`或`(x, y, z)`
- **大小限制**：各维度乘积不超过65535，即`x * y * z <= 65535`

#### 大shape算子的Grid处理策略

对于输入shape较大的算子，直接按照`BLOCK_SIZE`切分得到的grid总数可能超过65535，这在Triton-Ascend中是不支持的。有两种解决方案：

**方案1：kernel内循环处理（推荐）**

将grid设置得更小，让每个grid处理更多数据。由于UB（Unified Buffer）大小有限制，不能在一个grid中一次性处理所有分配的数据，而应该通过for循环分步分块处理：

```python
# 示例：处理大shape的向量操作
@triton.jit
def large_vector_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_GRID: tl.constexpr,  # 每个grid负责的元素总数
):
    pid = tl.program_id(0)
    
    # 计算当前grid负责的数据范围
    grid_start = pid * ELEMENTS_PER_GRID
    grid_end = min(grid_start + ELEMENTS_PER_GRID, n_elements)
    
    # 分块处理当前grid负责的数据
    for block_start in range(grid_start, grid_end, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < grid_end
        
        # 加载、计算、存储
        data = tl.load(input_ptr + offsets, mask=mask)
        result = compute_function(data)
        tl.store(output_ptr + offsets, result, mask=mask)

# 启动方式
def launch_large_kernel(input_tensor):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = * # 设置为尽可能大的合适的每次处理的值，充分利用ub
    MAX_GRID_SIZE = * # 设置为对应处理核的整数倍（区分vec与cube），小于65535的值
    
    # 计算每个grid需要处理的元素数
    ELEMENTS_PER_GRID = triton.cdiv(n_elements, MAX_GRID_SIZE)
    # 向上取整到BLOCK_SIZE的倍数，确保循环能完整处理
    ELEMENTS_PER_GRID = triton.cdiv(ELEMENTS_PER_GRID, BLOCK_SIZE) * BLOCK_SIZE
    
    # 计算实际需要的grid数量
    grid_size = triton.cdiv(n_elements, ELEMENTS_PER_GRID)
    
    output_tensor = torch.empty_like(input_tensor)
    large_vector_kernel[grid_size,](
        input_tensor, output_tensor, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        ELEMENTS_PER_GRID=ELEMENTS_PER_GRID,
    )
```

**方案2：host侧多次启动kernel**

如果一次内核启动无法处理整个张量，可以在host侧将张量分成多个部分，通过多次启动内核来完成计算：

```python
def launch_with_batching(x):
    M = x.shape[0]
    BLOCK_SIZE = * # 设置为尽可能大的合适的每次处理的值，充分利用ub
    MAX_GRID_SIZE = * # 设置为对应处理核的整数倍（区分vec与cube），小于65535的值
    
    if M > MAX_GRID_SIZE:
        # 使用多批次处理
        for start_row in range(0, M, MAX_GRID_SIZE):
            end_row = min(start_row + MAX_GRID_SIZE, M)
            batch_size = end_row - start_row
            
            # 提取当前批次的数据
            x_batch = x[start_row:end_row]
            
            # 启动内核处理当前批次
            grid = (batch_size,)
            op_kernel[grid](x_batch, x_batch.stride(0), BLOCK_SIZE)
    else:
        # 单次启动即可
        grid = (M,)
        op_kernel[grid](x, x.stride(0), BLOCK_SIZE)
```

## 5. 调试与排查清单

### 内存访问问题
- [ ] 所有load/store是否都有mask或boundary_check？
- [ ] stride参数设置是否正确？
- [ ] 数组索引是否越界？

### 控制流检查
- [ ] 是否误用了return/break/continue？
- [ ] 复杂条件是否用mask组合实现？
- [ ] tl.constexpr是否只在内核参数中使用？

### Grid与Block配置检查
- [ ] Grid总大小是否不超过65535？
- [ ] 对于大shape算子，是否采用了kernel内循环或host侧分批处理？
- [ ] Grid维度是否为tuple类型且不超过3维？

### 并发与原子操作检查
- [ ] 并发写入是否使用了原子操作？
- [ ] 原子操作的数据类型是否匹配？

### 切片与索引检查
- [ ] 是否避免了Python风格的直接切片（如`b[0]`、`b[i:j]`）？
- [ ] 是否对`tl.arange`生成的张量误用了`tl.get_element`？
- [ ] 切片操作是否使用了正确的API（`tl.get_element`、`tl.extract_slice`等）？

### 性能优化检查
- [ ] 内存访问是否连续（避免跨步访问）？
- [ ] 是否充分利用了块内并行？
- [ ] 复杂算子是否考虑拆分为多个简单kernel？

## 6. 常见错误速查

| 错误类型 | 典型症状 | 常见原因 | 解决方案 |
|---------|---------|---------|---------|
| 内存越界访问 | 运行时错误、结果异常、随机崩溃 | load/store缺少mask或boundary_check | 添加正确的mask或boundary_check保护 |
| Grid超限 | 编译失败或运行时错误 | grid总大小超过65535 | 使用kernel内循环或host侧分批处理 |
| 控制流错误 | 编译失败、语法错误 | 使用了return/break/continue | 移除禁用语句，使用mask控制流程 |
| 切片语法错误 | 编译失败 | 使用了`b[0]`或`b[i:j]`直接切片 | 使用`tl.get_element`或`tl.extract_slice` |
| tl.arange索引错误 | 编译失败 | 对`tl.arange`结果使用`get_element` | 直接计算索引值而非提取 |
| 类型转换错误 | 编译警告或错误 | 使用`tl.float16(scalar)`转换 | 改用`scalar.to(tl.float16)` |
| constexpr误用 | 编译失败 | 在host侧使用tl.constexpr | 仅在kernel参数中使用tl.constexpr |
| Stride设置错误 | 计算结果错误、数据错位 | stride参数计算或传递错误 | 验证stride设置，检查tensor.stride() |
| 数值不稳定 | 结果为NaN或Inf | softmax/sqrt等操作溢出 | 减去最大值、检查非负、使用float32 |
| 数据竞争 | 结果不确定、每次运行不同 | 多program并发写入同一位置 | 使用tl.atomic_add等原子操作 |
| BLOCK_SIZE过大 | 编译失败或运行时错误 | BLOCK_SIZE超过65536或硬件限制 | 减小BLOCK_SIZE，使用循环处理 |
| tl.where偏移计算 | 编译失败（Ascend后端） | 在内存偏移中使用tl.where | 改用if-else静态分支处理 |
| 性能低下 | 运行缓慢 | 内存访问不连续、切分不合理 | 优化内存布局、调整BLOCK_SIZE、使用block_ptr |
