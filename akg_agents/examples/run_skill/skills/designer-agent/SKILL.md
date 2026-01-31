---
name: designer-agent
description: "算法设计Agent，负责生成算子设计方案和优化策略"
level: L2
category: agent
version: "1.0.0"
license: MIT
---

# Designer Agent - 算法设计专家

## 角色定位

Designer Agent负责算法层面的设计，在代码生成之前提供：
- 算法框架设计
- 数据流分析
- 性能优化策略
- 伪代码/Sketch生成

## 核心能力

### 1. 算法分析

根据算子需求分析：
- 计算复杂度（时间/空间）
- 并行机会识别
- 数据依赖关系
- 内存访问模式

### 2. 设计方案生成

输出包含：
- 算法伪代码
- 数据分块策略
- 并行化方案
- 优化建议

### 3. 变种探索

在进化算法中生成多个设计变种：
- 不同的分块策略
- 不同的计算顺序
- 不同的内存层次利用

## 工作流程

```
输入: 算子规格 + 性能要求 + 硬件约束
  ↓
步骤1: 分析算子特征
  ├─ 计算密集 vs 访存密集
  ├─ 规则 vs 不规则
  └─ 独立 vs 依赖
  ↓
步骤2: 生成设计方案
  ├─ 选择合适的算法模式
  ├─ 确定分块策略
  └─ 规划内存使用
  ↓
步骤3: 输出设计文档
  ├─ 伪代码
  ├─ 数据流图
  └─ 优化建议
  ↓
输出: 设计Sketch → 交给Coder实现
```

## 设计模式库

### 1. Element-wise模式

适用于逐元素操作（ReLU, Sigmoid, 加法等）：

```
Design Pattern: Element-wise
- 并行化：每个线程处理一个或多个元素
- 内存：简单的顺序访问
- 优化：向量化加载，循环展开

Pseudocode:
  for each element in parallel:
      output[i] = f(input[i])
```

### 2. Reduction模式

适用于规约操作（Sum, Max, Min等）：

```
Design Pattern: Reduction
- 并行化：树状规约
- 内存：先local规约，再global规约
- 优化：使用shared memory，warp shuffle

Pseudocode:
  Step 1: Local reduction (per block)
      shared_mem[tid] = local_sum
      for offset in [N/2, N/4, ..., 1]:
          shared_mem[tid] += shared_mem[tid + offset]
  
  Step 2: Global reduction
      global_sum = atomicAdd(shared_mem[0])
```

### 3. Matrix Multiplication模式

```
Design Pattern: Tiled Matrix Multiplication
- 并行化：2D分块，每个block计算一个tile
- 内存：使用shared memory缓存tiles
- 优化：避免bank conflict，使用tensor cores

Pseudocode:
  for each block (bx, by):
      for tile_k in [0, K, TILE_SIZE]:
          Load A[bx, tile_k] to shared_A
          Load B[tile_k, by] to shared_B
          sync()
          
          Compute C_tile += shared_A @ shared_B
          sync()
      
      Write C_tile to C[bx, by]
```

### 4. Stencil模式

适用于需要相邻元素的操作（卷积、滤波等）：

```
Design Pattern: Stencil with Halo
- 并行化：每个block处理一个区域+halo
- 内存：加载halo到shared memory
- 优化：重用shared memory数据

Pseudocode:
  Load tile with halo to shared_mem
  sync()
  
  for each element in tile:
      result = 0
      for each neighbor in stencil:
          result += shared_mem[neighbor] * weight
      output[i] = result
```

## 设计决策树

### MatMul设计决策

```
MatMul设计
├─ 矩阵尺寸？
│   ├─ 小矩阵 (< 1024) → 简单实现，不分块
│   └─ 大矩阵 → 分块实现
│       ├─ 分块大小？
│       │   ├─ 32x32 (小，低occupancy)
│       │   ├─ 64x64 (中，平衡)
│       │   └─ 128x128 (大，高性能)
│       └─ 使用Tensor Cores？
│           ├─ 是 → wmma API (FP16/BF16)
│           └─ 否 → FMA指令 (FP32)
├─ 内存布局？
│   ├─ Row-major → 标准实现
│   └─ Col-major → 转置处理
└─ 稀疏性？
    ├─ 密集 → 标准算法
    └─ 稀疏 → CSR/COO格式
```

## 性能估算

### Roofline分析

```python
def estimate_performance(design):
    """估算设计的理论性能"""
    # 计算FLOP数
    flops = design.compute_operations()
    
    # 计算内存访问量
    bytes = design.memory_access()
    
    # 算术强度
    arithmetic_intensity = flops / bytes
    
    # 理论性能
    peak_flops = get_device_peak_flops()
    peak_bandwidth = get_device_peak_bandwidth()
    
    # Roofline
    compute_bound_perf = peak_flops
    memory_bound_perf = peak_bandwidth * arithmetic_intensity
    
    estimated_perf = min(compute_bound_perf, memory_bound_perf)
    
    return {
        'flops': flops,
        'bytes': bytes,
        'arithmetic_intensity': arithmetic_intensity,
        'estimated_gflops': estimated_perf / 1e9,
        'bottleneck': 'compute' if compute_bound_perf < memory_bound_perf else 'memory'
    }
```

## 设计文档模板

```markdown
# 算子设计: [算子名称]

## 1. 问题分析
- 输入：[形状，类型]
- 输出：[形状，类型]
- 计算：[描述]
- 复杂度：O(?)

## 2. 算法选择
- 模式：[Element-wise / Reduction / MatMul / Stencil / Custom]
- 理由：[为什么选择这个模式]

## 3. 并行化策略
- Grid dimension: [配置]
- Block dimension: [配置]
- 每个线程处理: [工作量]

## 4. 内存策略
- Global memory: [访问模式]
- Shared memory: [使用方案]
- Registers: [估算]

## 5. 优化策略
- [ ] 合并内存访问
- [ ] 使用shared memory
- [ ] 循环展开
- [ ] Warp-level优化
- [ ] Tensor Core使用

## 6. 伪代码
```
[详细的伪代码]
```

## 7. 性能估算
- FLOPS: [估算]
- Bandwidth: [估算]
- 瓶颈: [Compute/Memory]
- 预期性能: [GFLOPS]

## 8. 风险与挑战
- [可能的问题]
- [缓解措施]
```

## 进化策略

### 变异操作

```python
def mutate_design(parent_design, mutation_rate=0.3):
    """生成设计变种"""
    child_design = copy.deepcopy(parent_design)
    
    if random.random() < mutation_rate:
        # 变异1: 改变block size
        child_design.block_size = random.choice([128, 256, 512, 1024])
    
    if random.random() < mutation_rate:
        # 变异2: 改变tile size
        child_design.tile_size = random.choice([16, 32, 64, 128])
    
    if random.random() < mutation_rate:
        # 变异3: 改变计算顺序
        child_design.loop_order = random_permutation(['i', 'j', 'k'])
    
    return child_design
```

### 交叉操作

```python
def crossover_designs(parent1, parent2):
    """交叉两个设计"""
    child = Design()
    
    # 从parent1继承block配置
    child.block_size = parent1.block_size
    child.grid_size = parent1.grid_size
    
    # 从parent2继承内存策略
    child.tile_size = parent2.tile_size
    child.shared_mem_layout = parent2.shared_mem_layout
    
    return child
```

## 实际案例

### 案例1: Flash Attention设计

```
算子: Flash Attention
分析:
  - Attention计算: O(N^2) 复杂度
  - 瓶颈: 内存访问（需读写大矩阵多次）
  
设计方案:
  1. 分块计算（Tiling）
     - 将Q, K, V分成blocks
     - 每个block独立计算attention
  
  2. 在线Softmax
     - 避免存储完整的attention matrix
     - 使用online算法累积softmax
  
  3. 重计算策略
     - Forward不存储attention matrix
     - Backward时重新计算
  
优化:
  - Shared memory存储tiles
  - Fused kernel减少访存
  - 使用Tensor Cores加速矩阵乘法
  
结果:
  - 内存使用: O(N) vs 标准O(N^2)
  - 速度: 2-4x加速
```

### 案例2: Sparse MatMul设计

```
算子: Sparse Matrix Multiplication
分析:
  - 输入: 稀疏矩阵（CSR格式）+ 密集矩阵
  - 挑战: 不规则内存访问，负载不均衡
  
设计方案:
  1. 行并行策略
     - 每个warp处理一行
     - 使用warp-level reduction
  
  2. 负载均衡
     - 动态任务分配
     - 长行拆分到多个warps
  
  3. 内存优化
     - Prefetch稀疏矩阵索引
     - Cache密集矩阵的列
  
优化:
  - Warp shuffle通信
  - Vector load (float4)
  - Early exit for empty rows
  
结果:
  - vs 密集实现: 5-10x加速（高稀疏度）
  - vs cuSPARSE: 持平或略优
```

## 工具集成

### 与Coder Agent协作

```python
# Designer生成设计
design = designer.generate_design(
    operator_spec="matmul",
    shape=(1024, 1024, 1024),
    target_device="A100"
)

# Coder实现设计
code = coder.implement_design(
    design=design,
    dsl="triton",
    optimization_level=2
)

# 如果性能不达标，Designer调整设计
if performance < target:
    design = designer.refine_design(
        original_design=design,
        profiling_results=profile_data
    )
```

## 最佳实践

1. **先分析，后设计**：理解问题特征
2. **使用设计模式**：不要重新发明轮子
3. **估算性能**：设计前预测瓶颈
4. **迭代优化**：从简单到复杂
5. **文档化决策**：记录为什么这样设计

## 相关Skill

- **协作**: coder-agent (实现设计)
- **上游**: adaptive-evolve (进化工作流)
- **下游**: verifier-agent (验证设计效果)

