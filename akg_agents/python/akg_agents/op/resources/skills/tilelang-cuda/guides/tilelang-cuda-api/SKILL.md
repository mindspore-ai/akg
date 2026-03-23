---
name: tilelang-cuda-api
description: "TileLang CUDA API 完整参考手册，适用于需要查阅具体 API 用法、了解函数参数含义的任意 TileLang CUDA 内核代码生成场景"
category: fundamental
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA API 参考手册

本文档提供 TileLang 核心 API 的详细参考，包括函数签名、参数说明和使用示例。

## 1. 内核定义与编译

### @tilelang.jit(out_idx)
```python
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
- **作用**: 将 TileLang 函数编译为 GPU 内核
- **参数**: `out_idx` - 指定输出张量的索引列表（如 `[-1]` 表示最后一个参数为输出）
- **调用**: 设置 `out_idx` 后，运行内核时只需传入输入张量，输出由 TileLang 自动创建

### tilelang.compile
```python
kernel = tilelang.compile(my_func, out_idx=[-1])
```
- **作用**: 编译 TileLang 函数（等价于 `@tilelang.jit`）

## 2. 内核上下文

### T.Kernel(grid_x, grid_y, threads)
```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    # bx, by 对应 blockIdx.x, blockIdx.y
    pass
```
- **参数**:
  - `grid_x`: 网格 X 维度大小
  - `grid_y`: 网格 Y 维度大小（可选）
  - `threads`: 每个线程块的线程数
- **返回**: 线程块索引 `(bx, by)`

### T.ceildiv(a, b)
```python
grid_size = T.ceildiv(N, block_N)
```
- **参数**: `a`, `b` - 被除数和除数
- **返回**: 向上取整的除法结果
- **用途**: 计算网格大小

## 3. 内存分配 API

### T.alloc_shared(shape, dtype)
```python
A_shared = T.alloc_shared((block_M, block_K), "float16")
```
- **作用**: 分配共享内存（对应 GPU 共享内存）
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **用途**: 缓存频繁访问的数据

### T.alloc_fragment(shape, dtype)
```python
C_local = T.alloc_fragment((block_M, block_N), "float")
```
- **作用**: 分配寄存器片段（对应 GPU 寄存器文件）
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **用途**: 累加器和临时存储

### T.alloc_local(shape, dtype)
```python
temp = T.alloc_local((1,), "float32")
```
- **作用**: 分配线程本地内存
- **参数**:
  - `shape`: 张量形状
  - `dtype`: 数据类型
- **用途**: 线程私有的临时变量

## 4. 数据操作 API

### T.copy(src, dst)
```python
# 全局内存到共享内存
T.copy(A[by * block_M, ko * block_K], A_shared)

# 寄存器到全局内存
T.copy(C_local, C[by * block_M, bx * block_N])
```
- **作用**: 高效内存复制，自动合并访问
- **参数**:
  - `src`: 源数据（可以是全局内存切片或寄存器片段）
  - `dst`: 目标数据

### T.clear(tensor)
```python
T.clear(C_local)
```
- **作用**: 将张量清零
- **参数**: `tensor` - 要清零的张量

### T.fill(tensor, value)
```python
T.fill(buffer, -T.infinity("float"))
```
- **作用**: 用指定值填充张量
- **参数**:
  - `tensor`: 目标张量
  - `value`: 填充值

## 5. 循环控制 API

### T.Pipelined(count, num_stages)
```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.gemm(A_shared, B_shared, C_local)
```
- **作用**: 软件流水线循环，重叠内存操作和计算
- **参数**:
  - `count`: 循环次数
  - `num_stages`: 流水线深度（通常 2-4 效果最好）

### T.Parallel(dim1, dim2, ...)
```python
# 单维度并行
for i in T.Parallel(block_M):
    pass

# 多维度并行
for i, j in T.Parallel(block_M, block_N):
    C_local[i, j] = A_shared[i, j] + B_shared[i, j]
```
- **作用**: 并行循环，自动映射到线程
- **参数**: 各维度大小
- **优势**: 自动处理线程映射，避免手动线程索引计算错误

### T.serial(count)
```python
for k in T.serial(block_K):
    pass
```
- **作用**: 串行循环，顺序执行
- **参数**: `count` - 循环次数

### T.vectorized(count)
```python
for k in T.vectorized(TILE_K):
    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
```
- **作用**: 向量化循环，利用向量指令
- **参数**: `count` - 向量化长度

## 6. 内置计算原语

### T.gemm(A, B, C, transpose_B, policy)
```python
# 基础矩阵乘法
T.gemm(A_shared, B_shared, C_local)

# 带转置的矩阵乘法
T.gemm(Q_shared, K_shared, acc_s, transpose_B=True)

# 指定 Warp 策略
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
```
- **作用**: Tile 级别的矩阵乘法，利用 Tensor Core 加速
- **参数**:
  - `A`, `B`: 输入矩阵（通常为共享内存）
  - `C`: 输出/累加器（通常为寄存器片段）
  - `transpose_B`: 是否转置 B 矩阵
  - `policy`: Warp 策略

### T.reduce_max(input, output, dim, clear)
```python
T.reduce_max(acc_s, scores_max, dim=1)
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
```
- **作用**: 最大值归约
- **参数**:
  - `input`: 输入张量
  - `output`: 输出张量
  - `dim`: 归约维度
  - `clear`: 是否先清零输出（默认 True）

### T.reduce_sum(input, output, dim)
```python
T.reduce_sum(acc_s, scores_sum, dim=1)
```
- **作用**: 求和归约
- **参数**: 同 `T.reduce_max`

### T.reduce_min(input, output, dim)
```python
T.reduce_min(input_tensor, output_tensor, dim=axis)
```
- **作用**: 最小值归约

### T.reduce_mean(input, output, dim)
```python
T.reduce_mean(input_tensor, output_tensor, dim=axis)
```
- **作用**: 平均值归约

## 7. 数学函数

```python
T.exp(x)          # 指数函数
T.exp2(x)         # 以 2 为底的指数
T.log(x)          # 自然对数
T.sqrt(x)         # 平方根
T.rsqrt(x)        # 平方根倒数
T.infinity(dtype)  # 无穷大常量
```

## 8. 条件与逻辑操作

### T.if_then_else(condition, true_val, false_val)
```python
result = T.if_then_else(A[idx] > 0, A[idx] + B[idx], A[idx] - B[idx])
```
- **作用**: 条件选择（类似三元运算符）

### 条件分支
```python
for i in T.Parallel(block_M):
    if i < N:
        # 有条件的操作
        pass
```

## 9. 原子操作

### T.atomic_add(target, value)
```python
T.atomic_add(C_shared[tn], C_accum[0])
```
- **作用**: 线程安全的原子加法
- **参数**:
  - `target`: 目标内存位置
  - `value`: 要添加的值

## 10. 线程索引获取

### ✅ 推荐：T.Parallel
```python
for tn in T.Parallel(BLOCK_N):
    # tn 对应 threadIdx.x
    pass
```
- **优势**: 自动处理线程映射，更安全

### ⚠️ 不推荐：T.get_thread_binding
```python
# 可能导致问题，不推荐使用
tn = T.get_thread_binding(0)  # threadIdx.x
tk = T.get_thread_binding(1)  # threadIdx.y
```

## 11. 同步操作

### T.sync_threads()
```python
T.sync_threads()  # 所有线程都必须执行此操作
```
- **作用**: 线程块内同步
- **⚠️ 严格禁止**: 在条件分支中使用（会导致死锁）

## 12. 高级特性

### @T.macro
```python
@T.macro
def Softmax(acc_s, acc_s_cast, scores_max, scores_sum):
    T.reduce_max(acc_s, scores_max, dim=1)
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
    T.reduce_sum(acc_s, scores_sum, dim=1)
    T.copy(acc_s, acc_s_cast)
```
- **作用**: 定义可复用的宏操作

### T.annotate_layout / T.use_swizzle
```python
from tilelang.intrinsics import make_mma_swizzle_layout
T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
T.use_swizzle(panel_size=10, enable=True)
```
- **作用**: 内存布局优化和 L2 缓存光栅化

### 类型转换
```python
result = A[i].astype("float") * B[i].astype("float")
normalized.astype("float16")
```
- **作用**: 显式数据类型转换

## 数据类型支持

### 输入数据类型
- `float16`: 半精度浮点数
- `float32`: 单精度浮点数
- `bfloat16`: Brain Float 16
- `int8`: 8 位整数
- `int32`: 32 位整数

### 累积数据类型
- `float` / `float32`: 单精度浮点数（推荐用于累加器）

### 输出数据类型
- 通常与输入类型相同
- 可通过 `.astype()` 指定

## 使用建议

1. **选择合适的抽象级别**: Level 2 适合大多数应用，Level 3 用于极致性能优化
2. **合理的内存分配**: 共享内存用于频繁访问的数据，寄存器片段用于累加和临时存储
3. **优化数据移动**: 使用 `T.Parallel` 并行化数据复制，利用 `T.Pipelined` 重叠操作
4. **选择合适的线程数**: 通常为 128 或 256，考虑硬件特性和工作负载
5. **利用内置原语**: 使用 `T.gemm`、`T.reduce_sum` 等优化原语，避免重复实现已有功能

## 常见错误

1. **内存分配过大**: 超出硬件限制
2. **流水线深度不当**: 影响性能
3. **线程数不匹配**: 硬件利用率低
4. **数据类型不匹配**: 精度损失或性能下降
5. **⚠️ 同步使用错误**: 条件分支中的 `T.sync_threads()` 会导致死锁
6. **⚠️ 线程索引获取错误**: 使用 `T.get_thread_binding()` 而非 `T.Parallel()`
