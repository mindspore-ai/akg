---
name: tilelang-cuda-synchronization
description: "TileLang CUDA 同步规范，包括 T.sync_threads() 使用规则、线程安全最佳实践和死锁预防策略。适用于编写涉及共享内存访问、多线程协作、或需要避免同步死锁的 TileLang 内核代码生成场景"
category: implementation
version: "1.0.0"
metadata:
  backend: cuda
  dsl: tilelang_cuda
---

# TileLang CUDA 同步和线程安全

本文档详细说明 TileLang 中 `T.sync_threads()` 的使用规范和线程安全最佳实践。**同步问题是 TileLang 编程中最常见也是最危险的错误来源**。

---

## 1. T.sync_threads() 使用规范

### ⚠️ 严格禁止的使用场景

#### 1.1 条件分支中的同步

```python
# ❌ 错误示例 - 会导致死锁
if condition:
    T.sync_threads()  # 只有部分线程执行，其他线程永远等待

# ✅ 正确做法 - 所有线程都执行同步
T.sync_threads()
if condition:
    # 同步后的操作
```

#### 1.2 循环中的条件同步

```python
# ❌ 错误示例 - 死锁风险
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # 死锁风险

# ✅ 正确做法
for i in range(n):
    T.sync_threads()  # 所有线程都同步
    if tid < threshold:
        # 同步后的操作
```

#### 1.3 共享内存分配后的条件同步

```python
# ❌ 错误示例
if tid < N:
    shared_mem = T.alloc_shared((N,), dtype)
    T.sync_threads()  # 只有部分线程分配了共享内存

# ✅ 正确做法
shared_mem = T.alloc_shared((N,), dtype)
T.sync_threads()
if tid < N:
    # 使用共享内存
```

---

## 2. ❌ 绝对禁止：手动归约

手动归约是导致线程卡死的最常见原因。**必须使用内置归约函数**。

### 手动归约的危险

```python
# ❌ 绝对禁止：手动归约会导致线程卡死
while stride > 0:
    if tid < stride:
        shared[tid] += shared[tid + stride]
    T.sync_threads()  # 死锁风险，线程卡死
    stride //= 2

# ❌ 绝对禁止：条件分支中的同步
if condition:
    T.sync_threads()  # 死锁风险

# ❌ 绝对禁止：循环中的条件同步
for i in range(n):
    if tid < threshold:
        T.sync_threads()  # 死锁风险
```

### 症状识别
- **UTL（GPU 利用率）打满但 MEM 低**: 通常是线程卡死在同步点
- **内核永远不返回**: 死锁导致的无限等待
- **性能极差**: 不当的同步模式导致串行化

---

## 3. ✅ 正确的同步模式

### 3.1 在共享内存操作前后同步

```python
# ✅ 写入共享内存后同步
shared_mem[tid] = value
T.sync_threads()  # 确保所有写入完成

# ✅ 读取共享内存前同步
T.sync_threads()  # 确保所有写入完成
result = shared_mem[tid]
```

### 3.2 确保所有线程参与同步

```python
# ✅ 正确的同步模式
T.sync_threads()  # 所有线程都必须执行
# 后续操作
```

### 3.3 避免不必要的同步

```python
# ❌ 过度同步
for i in range(n):
    T.sync_threads()  # 每次迭代都同步，开销大

# ✅ 只在必要时同步
# 只在数据依赖需要时添加同步
```

---

## 4. ✅ 推荐的内置函数（替代手动同步）

### 4.1 内置归约函数

```python
# ✅ 推荐：内置归约函数，无需手动同步
T.reduce_sum(input_tensor, output_tensor, dim=axis)     # 求和归约
T.reduce_max(input_tensor, output_tensor, dim=axis)     # 最大值归约
T.reduce_min(input_tensor, output_tensor, dim=axis)     # 最小值归约
T.reduce_mean(input_tensor, output_tensor, dim=axis)    # 平均值归约
```

### 4.2 内置内存操作

```python
# ✅ 推荐：内置内存操作，自动处理同步
T.copy(src, dst)   # 高效的内存复制
T.clear(tensor)    # 清零操作
T.fill(tensor, value)  # 填充操作
```

### 4.3 内置数学函数

```python
# ✅ 推荐：内置数学函数
T.rsqrt(x)        # 平方根倒数
T.sqrt(x)         # 平方根
T.exp(x)          # 指数函数
T.log(x)          # 对数函数
```

---

## 5. 内置归约函数使用示例

### 5.1 LayerNorm 中的归约

```python
A_pow_local = T.alloc_fragment((M, N), "float32")
A_powsum = T.alloc_fragment((M,), "float32")

# ✅ 使用内置归约替代手动同步
T.reduce_sum(A_pow_local, A_powsum, dim=1)  # 按行求和
```

### 5.2 Softmax 中的归约

```python
A_exp = T.alloc_fragment((M, N), "float32")
A_sum = T.alloc_fragment((M,), "float32")
A_max = T.alloc_fragment((M,), "float32")

# ✅ 使用内置归约
T.reduce_max(A_exp, A_max, dim=1)    # 按行求最大值
T.reduce_sum(A_exp, A_sum, dim=1)    # 按行求和
```

### 5.3 完整的安全归约示例

```python
@tilelang.jit(out_idx=[-1])
def safe_reduction(M, N, block_size):
    @T.prim_func
    def main(x: T.Tensor((M, N), "float32"),
             y: T.Tensor((M,), "float32")):
        
        with T.Kernel(M, threads=block_size) as bx:
            # 分配寄存器片段
            x_local = T.alloc_fragment((N,), "float32")
            sum_local = T.alloc_fragment((1,), "float32")
            
            # 加载数据
            for tid in T.Parallel(N):
                x_local[tid] = x[bx, tid]
            
            # ✅ 安全的归约方式
            T.reduce_sum(x_local, sum_local, dim=0)
            
            # 写回结果
            y[bx] = sum_local[0]
    
    return main
```

---

## 6. 推荐的并行计算模式

### 6.1 使用 T.Parallel

```python
# ✅ 推荐：使用 T.Parallel 进行并行计算
for i, j in T.Parallel(M, N):
    result[i, j] = input[i, j] * scale[i]
```

### 6.2 使用 T.vectorized

```python
# ✅ 推荐：向量化操作
for i in T.vectorized(N):
    result[i] = input[i] * scale
```

### 6.3 使用 T.Pipelined

```python
# ✅ 推荐：软件流水线
for k in T.Pipelined(K, num_stages=3):
    T.copy(A[k], shared_A)
    T.gemm(shared_A, shared_B, result)
```

---

## 7. 线程索引获取规范

### ✅ 推荐：T.Parallel

```python
# ✅ 推荐方式：使用 T.Parallel 获取线程索引
for tn in T.Parallel(BLOCK_N):
    # tn 对应 threadIdx.x，自动处理线程映射
    pass
```

### ⚠️ 不推荐：T.get_thread_binding

```python
# ⚠️ 不推荐：可能导致问题
tn = T.get_thread_binding(0)  # threadIdx.x
tk = T.get_thread_binding(1)  # threadIdx.y
```

**为什么不推荐**：
- 可能导致线程映射错误
- 手动管理线程索引容易出错
- `T.Parallel` 提供更安全、更简洁的替代方案

---

## 8. 重要警告总结

### ❌ 绝对禁止
1. **手动归约**: 会导致线程卡死，UTL 打满但 MEM 低
2. **条件分支中的同步**: 会导致死锁
3. **循环中的条件同步**: 会导致死锁
4. **条件分支中的共享内存分配**: 会导致未定义行为

### ✅ 必须遵循
1. **使用内置归约函数**: `T.reduce_sum()`, `T.reduce_max()` 等
2. **确保所有线程参与同步**: `T.sync_threads()` 必须在所有线程执行路径上
3. **使用 T.Parallel 获取线程索引**: 替代 `T.get_thread_binding()`
4. **使用内置内存操作**: `T.copy()`, `T.clear()` 等

### 调试建议
- **UTL 高但 MEM 低**: 检查是否有手动归约导致的线程卡死
- **内核不返回**: 检查是否有条件分支中的同步导致死锁
- **结果错误**: 检查同步点是否正确，数据是否在同步后才使用
