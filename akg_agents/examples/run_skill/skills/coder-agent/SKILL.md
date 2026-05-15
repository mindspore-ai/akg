---
name: coder-agent
description: "代码生成Agent，负责将设计方案转换为可执行代码"
category: agent
version: "2.0.0"
license: MIT
structure:
  child_skills:
    - cuda-basics
    - triton-syntax
  default_children:
    - triton-syntax
---

# Coder Agent - 代码生成专家

## 角色定位

Coder Agent是 AKG Agents 中的核心代码生成组件，负责：
- 将算法设计转换为高性能代码
- 支持多种DSL（CUDA、Triton、OpenCL等）
- 支持多种硬件后端（NVIDIA、AMD、Intel等）
- 提供代码优化建议

## 核心能力

### 1. 多DSL支持

#### CUDA
- 完整的CUDA C++语法
- Kernel launch配置优化
- 内存管理（global, shared, register）
- 性能优化技巧

#### Triton
- Python-like语法
- 自动内存管理
- Block-level编程
- 编译器优化

#### OpenCL
- 跨平台支持
- 标准内核语法
- 平台特定优化

### 2. 多后端支持

| 后端 | 架构 | DSL优先级 |
|------|------|----------|
| NVIDIA GPU | CUDA | CUDA > Triton > OpenCL |
| AMD GPU | ROCm | OpenCL > HIP |
| Intel GPU | OneAPI | SYCL > OpenCL |

### 3. 代码优化

#### 内存优化
- 合并内存访问（Coalesced Access）
- 减少Bank Conflict
- 使用Shared Memory缓存
- Prefetching技术

#### 计算优化
- 循环展开（Loop Unrolling）
- 指令级并行（ILP）
- Warp级优化
- Tensor Core利用

#### 配置优化
- Block size调优
- Grid size计算
- Occupancy最大化
- Register压力控制

## 工作流程

```
输入: 算法设计 + 目标后端 + 性能要求
  ↓
步骤1: 加载相关Skill（如cuda-basics, triton-syntax）
  ↓
步骤2: 生成初始代码框架
  ↓
步骤3: 填充计算逻辑
  ↓
步骤4: 应用优化技巧
  ↓
步骤5: 添加错误处理
  ↓
输出: 可编译的高性能代码
```

## 代码生成策略

### 保守策略（Conservative）
- 优先正确性
- 使用标准模式
- 适合初次实现

### 迭代策略（Iterative）
- 先简单实现
- 逐步优化
- 适合复杂算子

### 激进策略（Aggressive）
- 直接使用高级优化
- 可能需要调试
- 适合性能关键场景

## 代码模板

### CUDA MatMul模板

```cuda
__global__ void matmul_kernel(
    const float* A, 
    const float* B, 
    float* C,
    int M, int N, int K
) {
    // 共享内存
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // 计算线程索引
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // 分块计算
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载数据到共享内存
        if (row < M && (tile * TILE_SIZE + threadIdx.x) < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + tile * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && (tile * TILE_SIZE + threadIdx.y) < K)
            Bs[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // 计算部分和
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 写回结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Triton MatMul模板

```python
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 程序ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 块偏移
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 指针
    a_ptrs = A_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    
    # 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 分块计算
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # 写回
    c_ptrs = C_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))
```

## RAG增强

Coder Agent使用RAG（Retrieval-Augmented Generation）从成功案例中学习：

1. **代码库检索**: 查找相似算子的实现
2. **模式识别**: 提取常见优化模式
3. **知识融合**: 将检索结果融入生成过程

## 错误处理

### 常见错误
1. **内存访问越界**: 添加边界检查
2. **Race Condition**: 使用同步原语
3. **数值精度问题**: 选择合适的数据类型
4. **性能退化**: 检查内存访问模式

### 调试技巧
```python
# 启用详细日志
coder.set_debug_mode(True)

# 生成带注释的代码
code = coder.generate(
    design=design,
    add_comments=True,
    add_assertions=True
)

# 性能分析
profile = coder.profile_code(code)
print(f"Expected performance: {profile['gflops']} GFLOPS")
```

## 配置示例

```yaml
coder_agent:
  dsl: triton  # cuda, triton, opencl
  backend: nvidia  # nvidia, amd, intel
  optimization_level: 2  # 0-3
  strategy: iterative  # conservative, iterative, aggressive
  enable_rag: true
  max_retries: 3
```

## 性能指标

| DSL | 编译时间 | 运行性能 | 可移植性 |
|-----|---------|---------|---------|
| CUDA | 中 | 优秀 | 低（NVIDIA专用）|
| Triton | 快 | 优秀 | 中（GPU通用）|
| OpenCL | 慢 | 良好 | 高（跨平台）|

## 最佳实践

1. **选择合适的DSL**: 
   - 单一NVIDIA后端 → CUDA
   - 跨GPU平台 → Triton
   - 跨设备类型 → OpenCL

2. **优化策略**:
   - 先正确性，后性能
   - 使用Profiler定位瓶颈
   - 应用针对性优化

3. **代码复用**:
   - 建立代码模板库
   - 提取通用组件
   - 使用RAG检索相似实现

4. **测试验证**:
   - 单元测试覆盖边界情况
   - 性能测试对比baseline
   - 正确性测试使用标准实现验证

## 相关Skill

- **子Skill**: cuda-basics, triton-syntax, optimization-techniques
- **协作**: verifier-agent (验证生成的代码)
- **上游**: designer-agent (提供算法设计)

