# 算子草图生成指导方案

## 概述

算子草图是介于算子需求和具体代码实现之间的中间设计层，用于指导大模型生成各种DSL的算子代码。它具有以下特点：
- **通用性**：不绑定特定硬件或DSL，可适配多种后端
- **清晰性**：结构化描述算法框架和计算逻辑
- **简洁性**：避免过度具体化，保持抽象层次的合理性
- **向量化友好**：确保算法设计充分利用向量化能力

---

# 第一部分：算子草图结构框架

## 1. 草图基本结构

每个算子草图包含以下核心部分：

```
算子草图 := {
    元信息定义,
    参数定义,
    并行策略,
    算法框架
}
```

### 1.1 元信息定义
- **算子名称**：明确算子功能
- **输入输出**：定义张量的形状和数据类型
- **计算特征**：标识算子类型（元素级、矩阵乘、归约等）

### 1.2 参数定义
- **硬编码参数**：来自 `get_init_inputs()` 的配置参数
- **运行时参数**：来自 `get_inputs()` 的输入张量
- **优化参数**：切分大小、循环次数等

### 1.3 并行策略
- **核间并行**：确定在哪个维度进行核间并行
- **数据分布**：每个核处理的数据范围
- **核内向量化**：确保每个核内充分利用向量化能力

### 1.4 算法框架
- **数据流**：load → compute → store 的基本流程
- **循环结构**：嵌套循环的组织方式
- **计算逻辑**：使用自然语言描述的具体计算过程

---

# 第二部分：通用语法框架

## 2. 数据操作语法

### 2.1 张量定义
```
tensor_name: Tensor[shape, dtype]
例：input_a: Tensor[(M, K), float32]
```

### 2.2 数据分块
```
tile_name = load(source_tensor[slice], size=tile_size)
例：a_tile = load(input_a[m_start:m_end, k_start:k_end], size=(64, 64))
```

### 2.3 数据存储
```
store(target_tensor[slice], tile_name)
例：store(output[m_start:m_end, n_start:n_end], c_tile)
```

## 3. 并行控制语法

### 3.1 核标识
```
core_id = get_core_id()
```

### 3.2 数据分配
```
data_range = partition(total_size, num_cores, core_id)
例：m_range = partition(M, num_cores, core_id)
```

### 3.3 流水线循环
```
for iteration in pipeline_range(loop_count):
    # 循环体
```

## 4. 计算描述语法

### 4.1 基础计算
使用自然语言描述，配合数学符号，标注待实现部分：
```
# 计算逻辑：矩阵乘法
# 对于输出矩阵C的每个元素C[i,j]：
# C[i,j] = sum(A[i,k] * B[k,j]) for k in range(K)
# 【待实现】：具体的tile级矩阵乘法计算
```

### 4.2 复杂算法
分步骤描述，标注待实现：
```
# 计算逻辑：softmax
# 步骤1：计算最大值 max_val = max(input[i] for i in range(N))
# 【待实现】：归约求最大值
# 步骤2：计算指数 exp_vals[i] = exp(input[i] - max_val)
# 【待实现】：向量化指数计算
# 步骤3：计算和 sum_exp = sum(exp_vals[i] for i in range(N))
# 【待实现】：归约求和
# 步骤4：归一化 output[i] = exp_vals[i] / sum_exp
# 【待实现】：向量化除法
```

---

# 第三部分：算子草图生成模板

## 3. 标准草图模板

```
# 算子名称：{operator_name}
# 功能描述：{function_description}

## 1. 元信息定义
- 输入：{input_tensors}
- 输出：{output_tensors}
- 计算类型：{compute_type}

## 2. 参数配置
### 硬编码参数
{init_parameters}

### 输入参数
{runtime_parameters}

### 优化参数
- tile_size: {tile_dimensions}
- num_cores: {core_count}
- loop_count: {iteration_count}

## 3. 并行策略
- 核间并行维度：{parallel_dimension}
- 数据分布：每个核处理 {data_per_core}
- 核内向量化策略：{vectorization_strategy}
- 边界处理：{boundary_handling}

## 4. 算法框架
### 4.1 核初始化
```
core_id = get_core_id()
data_range = partition({total_dimension}, num_cores, core_id)
```

### 4.2 主计算循环
```
for outer_loop in pipeline_range(outer_count):
    # 计算当前迭代的数据索引
    {index_calculation}
    
    # 数据加载
    {data_loading}
    
    for inner_loop in range(inner_count):
        # 内层计算逻辑
        {computation_logic}
        # 【待实现】：具体计算实现
    
    # 结果存储
    {data_storing}
```

### 4.3 计算逻辑详述
{detailed_computation_description}
# 【待实现】：各步骤的具体实现
```

---

# 第四部分：算法设计指导

## 4. 设计原则

### 4.1 核内向量化策略
- **向量化粒度**：确保每个核内处理向量块，而不是单个元素
- **数据对齐**：保证向量化访问的内存对齐
- **计算密度**：最大化每个向量操作的算术密度

### 4.2 数据局部性优化
- **空间局部性**：相邻数据一起加载和处理
- **时间局部性**：重复使用已加载的数据
- **缓存友好**：tile大小适配硬件缓存

### 4.3 核间并行化策略
- **负载均衡**：确保各核工作量相近
- **依赖最小化**：减少核间数据依赖
- **边界优化**：合理处理不能整除的情况

### 4.4 内存访问优化
- **连续访问**：优先使用连续内存访问模式
- **数据重用**：最大化数据在快速内存中的重用
- **流水线**：隐藏内存访问延迟

## 5. 常见模式

### 5.1 元素级操作（Element-wise）
```
特点：每个输出元素独立计算
核间并行策略：在任意维度进行核间分割
核内向量化策略：每个核内处理向量块
典型算子：ReLU, Add, Mul
```

### 5.2 归约操作（Reduction）
```
特点：多个输入元素计算一个输出
核间并行策略：在非归约维度分割
核内向量化策略：核内向量化归约，减少同步开销
典型算子：Sum, Max, Mean, Softmax
```

### 5.3 矩阵运算（Matrix Operations）
```
特点：涉及多维数据的复杂计算
核间并行策略：2D/3D tile分割
核内向量化策略：tile内向量化计算
典型算子：MatMul, Conv2D
```

### 5.4 数据重排（Data Rearrangement）
```
特点：改变数据的存储顺序
核间并行策略：输出导向的数据分割
核内向量化策略：向量化数据搬移
典型算子：Transpose, Reshape
```

---

# 第五部分：生成规范

## 6. 草图生成规范

### 6.1 必要元素
- [ ] 明确的输入输出定义
- [ ] 具体的参数配置
- [ ] 清晰的并行策略
- [ ] 完整的算法框架
- [ ] 详细的计算逻辑描述（标注待实现）

### 6.2 语言要求
- 使用**自然语言**描述计算逻辑
- 使用**伪代码**描述控制流
- 使用**数学公式**辅助说明
- 标注**【待实现】**的具体计算部分
- 保持**简洁明了**，避免过度具体化

### 6.3 通用性要求
- 不绑定特定DSL语法
- 不假设特定硬件特性
- 支持多种后端适配
- 便于自动化代码生成

## 7. 质量检查

### 7.1 完整性检查
- 是否覆盖所有输入输出
- 是否考虑边界条件
- 是否处理异常情况

### 7.2 正确性检查
- 算法逻辑是否正确
- 数据流是否完整
- 并行策略是否合理

### 7.3 优化检查
- 是否充分利用核内向量化能力
- 是否优化内存访问
- 是否考虑数据局部性

---

# 第六部分：示例草图

## 8. 示例：矩阵乘法草图

```
# 算子名称：MatrixMultiply
# 功能描述：计算两个矩阵的乘积 C = A × B

## 1. 元信息定义
- 输入：A: Tensor[(M, K), float32], B: Tensor[(K, N), float32]
- 输出：C: Tensor[(M, N), float32]
- 计算类型：矩阵运算

## 2. 参数配置
### 硬编码参数
- M = 1024, N = 1024, K = 1024

### 输入参数
- matrix_a: 输入矩阵A的数据指针
- matrix_b: 输入矩阵B的数据指针
- matrix_c: 输出矩阵C的数据指针

### 优化参数
- tile_m = 64, tile_n = 64, tile_k = 64
- num_cores: 32
- pipeline_depth: 3

## 3. 并行策略
- 核间并行维度：M维度（行方向）
- 数据分布：每个核处理 M/num_cores 行
- 核内向量化策略：tile内使用向量化计算，每个向量操作处理多个元素
- 边界处理：最后一个核处理剩余行

## 4. 算法框架
### 4.1 核初始化
```
core_id = get_core_id()
m_range = partition(M, num_cores, core_id)
m_start, m_end = m_range
```

### 4.2 主计算循环
```
for m_tile in pipeline_range((m_end - m_start + tile_m - 1) // tile_m):
    m_offset = m_start + m_tile * tile_m
    m_size = min(tile_m, m_end - m_offset)
    
    for n_tile in range((N + tile_n - 1) // tile_n):
        n_offset = n_tile * tile_n
        n_size = min(tile_n, N - n_offset)
        
        # 初始化输出tile
        c_tile = zeros(m_size, n_size)
        
        for k_tile in range((K + tile_k - 1) // tile_k):
            k_offset = k_tile * tile_k
            k_size = min(tile_k, K - k_offset)
            
            # 数据加载
            a_tile = load(matrix_a[m_offset:m_offset+m_size, k_offset:k_offset+k_size])
            b_tile = load(matrix_b[k_offset:k_offset+k_size, n_offset:n_offset+n_size])
            
            # 矩阵乘法计算
            # 计算逻辑：对于c_tile的每个元素(i,j)：
            # c_tile[i,j] += sum(a_tile[i,k] * b_tile[k,j]) for k in range(k_size)
            # 【待实现】：tile级矩阵乘法，使用向量化操作优化
            
        # 结果存储
        store(matrix_c[m_offset:m_offset+m_size, n_offset:n_offset+n_size], c_tile)
```

### 4.3 计算逻辑详述
矩阵乘法的核心计算：
1. 对于输出矩阵C的每个位置(i,j)
2. 计算A的第i行与B的第j列的点积
3. 点积计算：sum(A[i,k] * B[k,j]) for k in range(K)
4. 通过tile分块计算，累加各个k_tile的贡献
# 【待实现】：具体的GEMM计算，使用向量化指令优化
```

## 9. 示例：ReLU激活函数草图

```
# 算子名称：ReLU
# 功能描述：逐元素计算ReLU激活函数 output = max(0, input)

## 1. 元信息定义
- 输入：input: Tensor[(N,), float32]
- 输出：output: Tensor[(N,), float32]
- 计算类型：元素级操作

## 2. 参数配置
### 硬编码参数
- N = 10240

### 输入参数
- input_ptr: 输入数据指针
- output_ptr: 输出数据指针

### 优化参数
- tile_size = 256
- num_cores: 64

## 3. 并行策略
- 核间并行维度：N维度（数据长度方向）
- 数据分布：每个核处理 N/num_cores 个元素
- 核内向量化策略：每个核内处理向量块（如64个元素），使用向量化max操作
- 边界处理：最后一个核处理剩余元素

## 4. 算法框架
### 4.1 核初始化
```
core_id = get_core_id()
data_range = partition(N, num_cores, core_id)
start_idx, end_idx = data_range
```

### 4.2 主计算循环
```
for tile_idx in pipeline_range((end_idx - start_idx + tile_size - 1) // tile_size):
    current_start = start_idx + tile_idx * tile_size
    current_end = min(current_start + tile_size, end_idx)
    current_size = current_end - current_start
    
    # 数据加载
    input_tile = load(input_ptr[current_start:current_end])
    
    # ReLU计算
    # 计算逻辑：对于input_tile的每个元素x：
    # 如果 x > 0，则 output_element = x
    # 如果 x <= 0，则 output_element = 0
    # 数学表达式：output = max(0, input)
    # 【待实现】：向量化的max操作，每个向量处理多个元素
    
    # 结果存储
    store(output_ptr[current_start:current_end], output_tile)
```

### 4.3 计算逻辑详述
ReLU激活函数的逐元素计算：
1. 对于输入张量的每个元素x
2. 如果x大于0，输出x本身
3. 如果x小于等于0，输出0
4. 数学公式：f(x) = max(0, x)
# 【待实现】：使用向量化max指令，每个向量操作处理多个元素
```

## 10. 示例：Softmax归约操作草图

```
# 算子名称：Softmax
# 功能描述：计算Softmax归一化 output = exp(input - max) / sum(exp(input - max))

## 1. 元信息定义
- 输入：input: Tensor[(B, N), float32]
- 输出：output: Tensor[(B, N), float32]
- 计算类型：归约操作（在N维度归约）

## 2. 参数配置
### 硬编码参数
- B = 128, N = 1024

### 输入参数
- input_ptr: 输入数据指针
- output_ptr: 输出数据指针

### 优化参数
- tile_size = 256
- num_cores: 128 (每个batch分配一个核)

## 3. 并行策略
- 核间并行维度：B维度（batch方向）
- 数据分布：每个核处理一个batch
- 核内向量化策略：归约操作使用向量化算法，减少同步开销
- 归约维度：N维度（需要在核内归约）

## 4. 算法框架
### 4.1 核初始化
```
core_id = get_core_id()
batch_id = core_id  # 每个核处理一个batch
if batch_id >= B: return  # 边界检查
```

### 4.2 主计算循环
```
# 阶段1：计算最大值（归约）
max_val = -infinity
for tile_idx in pipeline_range((N + tile_size - 1) // tile_size):
    start_idx = tile_idx * tile_size
    end_idx = min(start_idx + tile_size, N)
    
    # 数据加载
    input_tile = load(input_ptr[batch_id, start_idx:end_idx])
    
    # 局部最大值计算
    # 【待实现】：向量化max归约，每个向量操作处理多个元素
    local_max = max(input_tile)
    max_val = max(max_val, local_max)

# 阶段2：计算指数和（归约）
sum_exp = 0.0
for tile_idx in pipeline_range((N + tile_size - 1) // tile_size):
    start_idx = tile_idx * tile_size
    end_idx = min(start_idx + tile_size, N)
    
    # 数据加载
    input_tile = load(input_ptr[batch_id, start_idx:end_idx])
    
    # 计算exp(x - max)
    # 【待实现】：向量化指数计算，每个向量操作处理多个元素
    exp_tile = exp(input_tile - max_val)
    
    # 局部求和
    # 【待实现】：向量化求和归约，每个向量操作处理多个元素
    local_sum = sum(exp_tile)
    sum_exp += local_sum
    
    # 暂存exp结果（用于最后的归一化）
    store_temp(exp_tile, tile_idx)

# 阶段3：归一化
for tile_idx in pipeline_range((N + tile_size - 1) // tile_size):
    start_idx = tile_idx * tile_size
    end_idx = min(start_idx + tile_size, N)
    
    # 加载暂存的exp结果
    exp_tile = load_temp(tile_idx)
    
    # 归一化
    # 【待实现】：向量化除法，每个向量操作处理多个元素
    output_tile = exp_tile / sum_exp
    
    # 结果存储
    store(output_ptr[batch_id, start_idx:end_idx], output_tile)
```

### 4.3 计算逻辑详述
Softmax的三阶段计算：
1. **最大值归约**：max_val = max(input[i]) for i in range(N)
2. **指数和归约**：sum_exp = sum(exp(input[i] - max_val)) for i in range(N)
3. **归一化**：output[i] = exp(input[i] - max_val) / sum_exp
# 【待实现】：
# - 高效的向量化归约算法
# - 向量化的指数计算
# - 向量化的除法操作
# - 数值稳定性处理（防止溢出）
```

---
