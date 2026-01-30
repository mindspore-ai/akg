# UnifiedSketch 设计

## 目标与原则

### 目标
用最小DSL表达算子设计意图，便于LLM理解和Coder实现。

### 原则
- **极简原语**：只有少数核心操作（alloc/load/store/compute/...）
- **统一语法**：所有操作都是函数调用风格，无语法差异
- **标准控制流**：使用Python for/range语法，不发明新语法
- **hint分离**：复杂优化用hint表达，不影响主逻辑清晰性

## 核心语法元素

### 结构声明
```python
sketch <op_name> {
  symbols: M, N, K;                    # 符号变量声明
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;  # 张量声明
  constexpr: m0, k0, n0
}
```

### @llm_hint 装饰器详解

#### 基本语法
`@llm_hint` 用于给 LLM 提供优化提示，帮助 coder 选择最优的实现策略。

```python
@llm_hint("optimization_type")              # 单一提示
@llm_hint("optimization_type", "context")   # 带上下文的提示
@llm_hint("opt1", "opt2", "opt3")          # 多重提示
```

#### 优化类型
- `"parallel"` - 并行化此循环
- `"pipeline"` - 流水线优化  
- `"vectorize"` - 向量化
- `"unroll"` - 循环展开

#### 硬件上下文提示
- `"grididx"` - GPU grid 级别并行（对应 blockIdx）
- `"threadidx"` - GPU thread 级别并行（对应 threadIdx）
- `"coreidx"` - NPU core 级别并行
- `"warp"` - GPU warp 级别优化
- `"simd"` - CPU/NPU SIMD 向量化

### for循环表达
```python
# GPU风格：grid + thread 两级并行
@llm_hint("parallel", "grididx.x")
for i in range(0, M, 128):                  # block级别
    @llm_hint("parallel", "threadidx.x") 
    for j in range(0, N, 32):               # thread级别
        @llm_hint("pipeline")
        for k in range(0, K, k_tile):
            # 计算逻辑

# NPU风格：core级别并行
@llm_hint("parallel", "coreidx")
for core_idx in range(num_cores):
    @llm_hint("pipeline")
    for k in range(0, K, k_tile):
        # 每个core的计算

# CPU风格：SIMD向量化
@llm_hint("parallel")                       # OpenMP并行
for i in range(0, M, tile_size):
    @llm_hint("vectorize", "simd")          # SIMD向量化
    for j in range(tile_size):
        # 向量化计算
```

### 核心操作
1. **alloc** - 内存分配
2. **load** - 数据加载  
3. **store** - 数据存储
4. **compute函数** - 计算操作

## 语法概览

```python
sketch matmul {
  symbols: M, N, K;
  tensors: A[M, K]: f16; B[K, N]: f16; C[M, N]: f32;
  
  m0, k0, n0 = 128, 256, 256

  @llm_hint("parallel")
  for i_outer in range(0, ceil(M, m0)):
    @llm_hint("parallel")
    for j_outer in range(0, ceil(N, n0)):

      # 内存分配
      c_tile = alloc([m0, n0], llm_hint=["accumulator", "init_zero"])
      a_tile = alloc([m0, k0], llm_hint=["fast", "input_cache"])
      b_tile = alloc([k0, n0], llm_hint=["fast", "input_cache"])

      @llm_hint("pipeline")
      for k_outer in range(0, ceil(K, k0)):
        # 数据搬移
        load(A[i_outer:i_outer+m0, k_outer:k_outer+k0] -> a_tile)
        load(B[k_outer:k_outer+k0, j_outer:j_outer+n0] -> b_tile)
        
        # 计算操作
        gemm(a_tile, b_tile, dst=c_tile)
        
      # 数据写回
      store(c_tile -> C[i_outer:i_outer+m0, j_outer:j_outer+n0])
}
```

## 内存管理系统

### alloc() 语法
```python
tile = alloc([shape], llm_hint=["存储要求", "用途说明", "性能要求"])
```

### hint设计原则
**语义化描述，让LLM根据硬件文档选择具体实现**

#### 存储要求（性能层次）
- `"fastest"` - 最快访问速度，容量小（让LLM选择register/L0等）
- `"fast"` - 快速访问，中等容量（让LLM选择shared/L1等）
- `"medium"` - 中等速度，较大容量（让LLM选择L2/cache等）
- `"slow"` - 较慢但容量大（让LLM选择global/DDR等）

#### 用途说明（帮助LLM理解意图）
- `"accumulator"` - 累加器，需要频繁读写
- `"input_cache"` - 输入数据缓存，主要读取
- `"output_buffer"` - 输出缓冲，主要写入
- `"temp_workspace"` - 临时工作空间
- `"shared_between_threads"` - 线程间共享数据

#### 初始化要求
- `"init_zero"` - 初始化为0
- `"no_init"` - 不初始化（默认）

### 示例
```python
# 语义化hint方式（推荐）
c_acc = alloc([128, 128], llm_hint=["fastest", "accumulator", "init_zero"])
a_cache = alloc([128, 256], llm_hint=["fast", "input_cache"])  
temp = alloc([128], llm_hint=["fast", "temp_workspace"])

# LLM会根据硬件文档将其映射为：
# NPU: fastest→L0, fast→L1_buffer
# GPU: fastest→register, fast→shared_memory  
# CPU: fastest→register, fast→L1_cache
```

## 数据搬移操作

### load() 语法
```python
load(tensor[slice] -> tile)
```

### store() 语法  
```python
store(tile -> tensor[slice])
```

### 切片表达
```python
# 基本切片
A[i:i+128, k:k+256]           # 二维切片
X[start:end]                  # 一维切片

# 完整tile
A[i_outer:i_outer+m0, k_outer:k_outer+k0]
```

### 示例
```python
load(A[0:128, 0:256] -> a_tile)              # 加载A的子块到a_tile
store(result_tile -> C[i:i+128, j:j+128])    # 将结果写回C的子块
```

## 计算操作库

### 基础运算
```python
add(src1, src2, dst)          # dst = src1 + src2
mul(src1, src2, dst)          # dst = src1 * src2  
sub(src1, src2, dst)          # dst = src1 - src2
div(src1, src2, dst)          # dst = src1 / src2
max(src1, src2, dst)          # dst = max(src1, src2)
min(src1, src2, dst)          # dst = min(src1, src2)
...
```

### 数学函数
```python
exp(src, dst)                 # dst = exp(src)
log(src, dst)                 # dst = log(src)
sqrt(src, dst)                # dst = sqrt(src)
abs(src, dst)                 # dst = abs(src)
tanh(src, dst)                # dst = tanh(src)
sigmoid(src, dst)             # dst = sigmoid(src)
...
```

### 线性代数
```python
gemm(a, b, dst)               # dst += a @ b (矩阵乘法)
dot(a, b, result)             # result = dot(a, b) (向量点积)
reduce_sum(src, axis, dst)    # dst = sum(src, axis=axis) (允许axis为list，指代多轴同时reduce)
reduce_max(src, axis, dst)    # dst = max(src, axis=axis)
...
```

### 复合函数
```python
relu(src, dst)                # dst = max(0, src)
gelu(src, dst)                # dst = gelu(src)
silu(src, dst)                # dst = silu(src) = src * sigmoid(src)
softmax(src, dst)             # dst = softmax(src)
...
```

## 并行与优化提示

### 多参数 @llm_hint 用法

#### 不同硬件的并行模式
```python
# GPU: 使用 grid + thread 两级并行
@llm_hint("parallel", "grididx")      # 对应 blockIdx.x/y/z
@llm_hint("parallel", "threadidx")    # 对应 threadIdx.x/y/z

# NPU: 使用 core 级别并行
@llm_hint("parallel", "coreidx")      # 对应 ai_core 并行

# CPU: 使用线程并行 + SIMD
@llm_hint("parallel")                 # 对应 OpenMP/TBB 
@llm_hint("vectorize", "simd")        # 对应 AVX/NEON
```

#### 组合使用策略
```python
# GPU完整示例
@llm_hint("parallel", "grididx")
for block_i in range(M_blocks):
    @llm_hint("parallel", "threadidx") 
    for thread_j in range(threads_per_block):
        @llm_hint("pipeline")
        for k in range(k_blocks):
            # 计算逻辑

# NPU完整示例  
@llm_hint("parallel", "coreidx")
for core_idx in range(num_cores):
    @llm_hint("pipeline")
    for k in range(k_tiles):
        @llm_hint("vectorize")
        for i in range(vector_size):
            # 向量计算
```

## 常见模式示例

### MatMul（如上面语法概览）

### Elementwise - ReLU
```python
sketch relu {
  symbols: N;
  tensors: X[N]: f32; Y[N]: f32;
  
  tile_size = 1024
  
  @llm_hint("parallel")
  for i in range(0, ceil(N, tile_size)):
    x_tile = alloc([tile_size], llm_hint="l1_buffer")
    y_tile = alloc([tile_size], llm_hint="l1_buffer")
    
    load(X[i:i+tile_size] -> x_tile)
    relu(x_tile, y_tile)
    store(y_tile -> Y[i:i+tile_size])
}
```

### Reduction - Softmax
```python
sketch softmax {
  symbols: B, N;
  tensors: X[B, N]: f32; Y[B, N]: f32;
  
  @llm_hint("parallel")
  for b in range(B):
    x_row = alloc([N], llm_hint="l1_buffer")
    y_row = alloc([N], llm_hint="l1_buffer")
    max_val = alloc([1], llm_hint="l0c")
    sum_val = alloc([1], llm_hint="l0c")
    
    load(X[b, 0:N] -> x_row)
    
    # 三阶段softmax
    reduce_max(x_row, axis=0, max_val)
    sub(x_row, max_val, x_row)        # x = x - max
    exp(x_row, y_row)                 # y = exp(x)
    reduce_sum(y_row, axis=0, sum_val)
    div(y_row, sum_val, y_row)        # y = y / sum
    
    store(y_row -> Y[b, 0:N])
}
```

### 复合算子 - GELU
```python
sketch gelu {
  symbols: N;
  tensors: X[N]: f32; Y[N]: f32;
  
  tile_size = 512
  
  @llm_hint("parallel")
  for i in range(0, ceil(N, tile_size)):
    x_tile = alloc([tile_size], llm_hint="l1_buffer")
    y_tile = alloc([tile_size], llm_hint="l1_buffer")
    
    load(X[i:i+tile_size] -> x_tile)
    gelu(x_tile, y_tile)              # 让coder决定如何实现
    store(y_tile -> Y[i:i+tile_size])
}
```

## 最佳实践

### 编写顺序
1. **先写基本结构**：symbols, tensors, 主循环框架
2. **再加内存管理**：alloc合适的tile
3. **然后加数据流**：load -> compute -> store
4. **最后加优化hint**：@llm_hint装饰器

### Tile大小设置
- 考虑硬件内存约束（如NPU UB大小、GPU shared memory限制）
- 优先选择2的幂次（128, 256, 512, 1024）
- 保证数据对齐要求

### 错误避免
- **不要混用抽象层次**：要么用高级函数（gelu），要么用基础运算（add+mul）
- **明确数据流向**：每个load都要有对应的compute，每个compute都要有对应的store
- **合理使用hint**：不要过度优化，先保证逻辑正确
