# TileLang API 参考文档

## 概述

TileLang 是一个专为高性能 GPU/CPU 内核开发设计的领域特定语言（DSL）。它采用类似 Python 的语法，底层基于 TVM 编译器基础设施，让开发者能够专注于生产力而不牺牲实现最优性能所需的底层优化。

## 核心特性

- **三层抽象级别**：从硬件无关到硬件感知的完整编程接口
- **显式内存管理**：直接控制硬件内存层次结构
- **软件流水线**：自动优化内存操作与计算的并行执行

## 编程接口层次

### Level 1: 硬件无关接口（Hardware-Unaware）
- **目标用户**：不需要了解特定硬件细节的用户
- **特点**：编译器自动处理内存层次和硬件特定优化
- **适用场景**：快速原型开发，专注于算法逻辑

### Level 2: 硬件感知接口 + Tile 库（Hardware-Aware with Tile Library）
- **目标用户**：对 GPU 内存层次有基本理解的开发者
- **特点**：提供预定义的 Tile 库操作和模式，无需深入线程级别的细节
- **适用场景**：大多数高性能计算应用，平衡了易用性和性能控制

### Level 3: 硬件感知接口 + 线程原语（Hardware-Aware with Thread Primitives）
- **目标用户**：对底层硬件特性有深入理解的专家
- **特点**：提供线程原语和低级构造的直接访问，允许针对特定 GPU 架构的精细优化
- **适用场景**：极致性能优化，需要精细控制硬件行为

## 核心 API

### 内核定义

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[-1])
def my_kernel(M, N, K, block_M, block_N, block_K):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # 内核实现
        pass
    return main
```

### 内核上下文

```python
# 初始化内核上下文
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx, by 对应 blockIdx.x, blockIdx.y
    # 内核逻辑
```

**参数说明**：
- `T.ceildiv(N, block_N)`：网格 X 维度（向上取整除法）
- `T.ceildiv(M, block_M)`：网格 Y 维度
- `threads=128`：每个线程块的线程数
- `(bx, by)`：线程块索引

### 内存分配

#### 共享内存分配
```python
# 共享内存分配 - 对应 GPU 的共享内存
A_shared = T.alloc_shared((block_M, block_K), "float16")
B_shared = T.alloc_shared((block_K, block_N), "float16")
```

#### 寄存器片段分配
```python
# 寄存器片段分配 - 对应 GPU 的寄存器文件
C_local = T.alloc_fragment((block_M, block_N), "float")
```

#### 本地内存分配
```python
# 本地内存分配 - 对应线程本地存储
temp_buffer = T.alloc_local((size,), dtype)
```

### 数据操作

#### 数据复制
```python
# 全局内存到共享内存
T.copy(A[by * block_M, ko * block_K], A_shared)

# 共享内存到全局内存
T.copy(C_local, C[by * block_M, bx * block_N])
```

#### 并行数据操作
```python
# 并行复制
for k, j in T.Parallel(block_K, block_N):
    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
```

#### 内存初始化
```python
T.clear(C_local)  # 清零
T.fill(buffer, value)  # 填充指定值
```

### 软件流水线

```python
# 流水线循环 - 重叠内存操作和计算
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    # 流水线阶段
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```

**参数说明**：
- `num_stages=3`：流水线深度，通常 2-4 个阶段效果最好

### 并行化控制

#### 并行循环
```python
# 并行循环 - 自动映射到线程
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```

#### 串行循环
```python
# 串行循环 - 顺序执行
for k in T.serial(block_K):
    # 串行操作
    pass
```

#### 向量化循环
```python
# 向量化循环 - 利用向量指令
for k in T.vectorized(TILE_K):
    # 向量化操作
    pass
```

### 内置计算原语

#### 矩阵乘法
```python
# Tile 级别的矩阵乘法
T.gemm(A_shared, B_shared, C_local)

# 带策略的矩阵乘法
T.gemm(A_shared, B_shared, C_local, 
       policy=T.GemmWarpPolicy.FullRow)  # 使用 Tensor Core
```

#### 归约操作
```python
# 最大值归约
T.reduce_max(acc_s, scores_max, dim=1)

# 求和归约
T.reduce_sum(acc_s, scores_sum, dim=1)
```

### 高级特性

#### 宏定义
```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum):
    T.reduce_max(acc_s, scores_max, dim=1)
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
    T.reduce_sum(acc_s, scores_sum, dim=1)
    T.copy(acc_s, acc_s_cast)
```

#### 条件执行
```python
# 条件分支
for i in T.Parallel(block_M):
    scores_max[i] = T.if_then_else(
        scores_max[i] == -T.infinity("float"), 
        0, 
        scores_max[i]
    )
```

#### 原子操作
```python
# 原子加法
T.atomic_add(C_shared[tn], C_accum[0])

# ⚠️ 线程同步 - 使用规范
# 确保所有线程都参与同步，避免死锁
T.sync_threads()  # 所有线程都必须执行此操作
```

#### 线程索引获取
```python
# ✅ 推荐方式：使用 T.Parallel 获取线程索引
for tn in T.Parallel(BLOCK_N):
    # tn 对应 threadIdx.x
    pass

# ⚠️ 不推荐：使用 T.get_thread_binding（可能导致问题）
# tn = T.get_thread_binding(0)  # threadIdx.x
# tk = T.get_thread_binding(1)   # threadIdx.y
```

### 同步和线程安全

#### T.sync_threads() - 线程同步
```python
# ⚠️ 严格禁止的使用场景
if condition:
    T.sync_threads()  # ❌ 死锁：只有部分线程执行同步

# ✅ 正确的使用方式
T.sync_threads()  # 所有线程都执行同步
if condition:
    # 同步后的操作
```

**使用规范：**
- 确保所有线程都参与同步
- 避免在条件分支中使用同步
- 在共享内存操作前后进行同步

#### T.Parallel() - 并行循环和线程索引
```python
# ✅ 推荐：使用 T.Parallel 获取线程索引
for tid in T.Parallel(threads):
    # tid 是线程索引
    shared_mem[tid] = value

# ✅ 并行计算
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```

**优势：**
- 自动处理线程映射
- 避免手动线程索引计算错误
- 更好的代码可读性

## 数据类型支持

### 输入数据类型
- `float16`：半精度浮点数
- `float32`：单精度浮点数
- `bfloat16`：Brain Float 16
- `int8`：8位整数
- `int32`：32位整数

### 累积数据类型
- `float`：单精度浮点数（推荐）
- `float32`：单精度浮点数

### 输出数据类型
- 通常与输入类型相同
- 可通过类型转换指定

## 最佳实践

### API 使用建议

1. **选择合适的抽象级别**：
   - Level 2 适合大多数应用
   - Level 3 用于极致性能优化

2. **合理的内存分配**：
   - 共享内存用于频繁访问的数据
   - 寄存器片段用于累加和临时存储

3. **优化数据移动**：
   - 使用 `T.Parallel` 并行化数据复制
   - 利用软件流水线重叠操作

4. **选择合适的线程数**：
   - 通常为 128 或 256
   - 考虑硬件特性和工作负载

5. **利用内置原语**：
   - 使用 `T.gemm` 等优化原语
   - 避免重复实现已有功能

### 常见错误

1. **内存分配过大**：超出硬件限制
2. **流水线深度不当**：影响性能
3. **线程数不匹配**：硬件利用率低
4. **数据类型不匹配**：精度损失或性能下降
5. **⚠️ 同步使用错误**：条件分支中的 `T.sync_threads()` 会导致死锁
6. **⚠️ 线程索引获取错误**：使用 `T.get_thread_binding()` 而非 `T.Parallel()`
