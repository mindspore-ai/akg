# TileLang-Ascend 编程基础

本文档介绍TileLang-Ascend的核心概念和编程模式，专为华为Ascend NPU设计。

## 1. 核心概念

### 内核定义
```python
@T.prim_func
def main(  # 【重要】函数名必须为main
    A: T.Tensor((M, N), "float32"),  # 使用元组定义shape
    B: T.Tensor((M, N), "float32"),
    C: T.Tensor((M, N), "float32"),
    shape: T.int32,
):
    # 内核实现
```
- **装饰器**: `@T.prim_func` 定义NPU内核函数
- **函数名**: 必须命名为`main`（NPUIR要求）
- **Tensor类型**: 使用`T.Tensor(shape_tuple, dtype)`声明输入输出张量
- **Shape格式**: 必须使用**元组**`(M, N)`，不能用列表`[M, N]`

### NPU内核启动

**关键约束**:
- 必须设置`is_npu=True`
- **只支持一维网格**（不同于CUDA的多维网格）
- **Grid Size必须是编译时确定的Python整数**（最重要！）

**正确模式** - 在闭包外部计算grid size:
```python
def create_kernel(N, block_N, dtype="float32"):
    # ✅ 在外部用Python计算（编译时确定）
    n_num = N // block_N
    
    @T.prim_func
    def main(A: T.Tensor((N), dtype), ..., shape: T.int32):
        # ✅ 直接使用外部的n_num
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # cid: 块索引 (block index)
            # shape: 运行时尺寸参数，用于边界检查
            
            # ✅ 运行时边界计算
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
```

**错误模式** - 在`@T.prim_func`内部计算:
```python
def create_kernel(N, block_size, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((N), dtype), ...):
        # ❌ 在prim_func内部计算 - 这是TVM表达式！
        n_blocks = T.ceildiv(N, block_size)
        
        with T.Kernel(n_blocks, is_npu=True) as (cid, _):
            # ❌ 导致: 'NoneType' object has no attribute 'group'
```

**为什么必须这样？**
- 编译器需要生成`T.launch_thread("blockIdx.x", <static_number>)`
- `T.Kernel()`的参数必须是Python整数，不能是TVM表达式
- 编译时确定并行度，运行时只处理边界

### NPU内存层次

| TIR作用域 | Ascend内存 | 用途 | 分配API |
|-----------|------------|------|---------|
| `"global"` | GM | 全局内存 | 张量参数 |
| `"shared.dyn"` | L1 | 一级缓存 | `T.alloc_L1()` |
| `"shared"` | UB | 统一缓冲区 | `T.alloc_ub()` |
| `"wmma.matrix_a"` | L0A | 矩阵A缓存 | `T.alloc_L0A()` |
| `"wmma.matrix_b"` | L0B | 矩阵B缓存 | `T.alloc_L0B()` |
| `"wmma.accumulator"` | L0C | 累加器 | `T.alloc_L0C()` |

**内存速度**: L0 (最快) > L1 > UB > GM (最慢)

### NPU双核心架构与作用域规则

**Cube核心** (矩阵运算):
```python
with T.Scope("Cube"):
    T.npuir_dot(A_l1, B_l1, C_l0c, initC=True)
```

**Vector核心** (向量运算):
```python
with T.Scope("Vector"):
    T.npuir_add(A_ub, B_ub, C_ub)
```

**作用域规则** (关键！):
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    # ===== T.Kernel作用域 =====
    # ✅ 可以在这里执行:
    # - 分配buffer: T.alloc_ub(), T.alloc_L1()等
    # - 计算索引: offset = cid * block_size
    # - 定义标量: scalar = 2.5
    
    x_ub = T.alloc_ub([block_size], dtype)
    offset = cid * block_size
    
    # ===== T.Scope作用域 =====
    with T.Scope("Vector"):  # 或 "Cube"
        # ✅ 必须在这里执行:
        # - 数据传输: T.copy()
        # - 所有计算: T.npuir_*()
        
        T.copy(X[offset], x_ub, [size])
        T.npuir_add(x_ub, y_ub, z_ub)
        T.copy(z_ub, Z[offset], [size])
```

**错误示例**:
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    x_ub = T.alloc_ub([block_size], dtype)
    T.copy(X[offset], x_ub, [size])  # ❌ 错误！在Scope外调用
    # 导致: TVMError: x_ub should be a memref
```

## 2. 标准编程模式

### 2.1 向量操作模式

```python
@T.prim_func
def main(  # 函数名为main
    A: T.Tensor((N,), "float32"),  # 使用元组
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
    N: T.int32
):
    block_size = T.int32(256)
    n_blocks = T.ceildiv(N, block_size)
    
    with T.Kernel(n_blocks, is_npu=True) as (cid, _):
        # 分配UB缓冲区
        A_ub = T.alloc_ub([block_size], "float32")
        B_ub = T.alloc_ub([block_size], "float32")
        C_ub = T.alloc_ub([block_size], "float32")
        
        # 计算偏移和边界
        offset = cid * block_size
        tail_size = T.min(block_size, N - offset)
        
        # 数据搬移和计算（必须在Vector scope中）
        with T.Scope("Vector"):
            T.copy(A[offset], A_ub, [tail_size])
            T.copy(B[offset], B_ub, [tail_size])
            T.npuir_add(A_ub, B_ub, C_ub)
            T.copy(C_ub, C[offset], [tail_size])
```

### 2.2 矩阵乘法模式

```python
@T.prim_func
def main(  # 函数名为main
    A: T.Tensor((M, K), "float16"),  # 使用元组
    B: T.Tensor((K, N), "float16"),
    C: T.Tensor((M, N), "float32"),
    M: T.int32, N: T.int32, K: T.int32
):
    tile_m, tile_k, tile_n = 16, 256, 16
    n_blocks = T.ceildiv(M, tile_m) * T.ceildiv(N, tile_n)
    
    with T.Kernel(n_blocks, is_npu=True) as (cid, _):
        # 计算块索引
        block_m = cid // T.ceildiv(N, tile_n)
        block_n = cid % T.ceildiv(N, tile_n)
        
        # 分配内存
        A_l1 = T.alloc_L1([tile_m, tile_k], "float16")
        B_l1 = T.alloc_L1([tile_k, tile_n], "float16")
        C_l0c = T.alloc_L0C([tile_m, tile_n], "float32")
        
        # K维度循环
        with T.Scope("Cube"):
            for k in T.serial(T.ceildiv(K, tile_k)):
                # 加载数据（ND→NZ格式）
                T.npuir_load_nd2nz(
                    A[block_m*tile_m, k*tile_k], 
                    A_l1, [tile_m, tile_k]
                )
                T.npuir_load_nd2nz(
                    B[k*tile_k, block_n*tile_n], 
                    B_l1, [tile_k, tile_n]
                )
                
                # 矩阵乘法
                init = (k == 0)
                T.npuir_dot(A_l1, B_l1, C_l0c, initC=init)
            
            # 存储结果（NZ→ND格式）
            T.npuir_store_fixpipe(
                C_l0c, C[block_m*tile_m, block_n*tile_n],
                size=[tile_m, tile_n], enable_nz2nd=True
            )
```

### 2.3 归约模式

```python
@T.prim_func
def main(  # 函数名为main
    X: T.Tensor((M, N), "float32"),  # 使用元组
    Y: T.Tensor((M,), "float32"),
    M: T.int32, N: T.int32
):
    with T.Kernel(M, is_npu=True) as (row_id, _):
        # 【重要】分配2D缓冲区（NPUIR要求所有buffer统一rank）
        x_ub = T.alloc_ub([1, N], "float32")    # [1, N]
        sum_ub = T.alloc_ub([1, 1], "float32")  # [1, 1]
        
        with T.Scope("Vector"):
            # 加载一行数据
            T.copy(X[row_id, 0], x_ub[0, 0], [1, N])
            
            # 归约求和（沿第二维度）
            T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")
            
            # 存储结果
            T.copy(sum_ub[0, 0], Y[row_id], [1])
```

## 3. 编译时 vs 运行时概念

### 关键区别

TileLang编程的核心在于理解**编译时确定**和**运行时计算**的区别：

| 概念 | 位置 | 类型 | 示例 | 用途 |
|------|------|------|------|------|
| **编译时参数** | 闭包外部 | Python整数 | `n_num = N // block_N` | Grid size, 并行度 |
| **运行时参数** | `@T.prim_func`参数 | `T.int32` | `shape: T.int32` | 动态尺寸，边界检查 |
| **TVM表达式** | `@T.prim_func`内部 | TVM Expr | `T.ceildiv(N, block)` | 运行时计算 |

**完整示例**:
```python
def create_kernel(N, block_N, dtype="float32"):
    # ===== 编译时区域 =====
    n_num = N // block_N  # Python整数，编译时确定
    
    @T.prim_func
    def main(
        A: T.Tensor((N), dtype),  # N是编译时常量
        B: T.Tensor((N), dtype),
        C: T.Tensor((N), dtype),
        shape: T.int32,  # shape是运行时参数
    ):
        # ===== 运行时区域 =====
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            # n_num: 编译时确定的Python值
            # cid: 运行时的块索引
            # shape: 运行时的实际尺寸
            
            # 运行时边界计算
            offset = cid * block_N  # block_N是编译时常量
            remaining = shape - offset  # shape是运行时值
            tail_size = T.min(block_N, remaining)
```

**为什么要这样设计？**
1. **并行度必须编译时确定**: NPU需要在编译时知道启动多少个block
2. **边界可以运行时检查**: 实际数据大小可能在运行时变化
3. **性能优化**: 编译时确定的值可以做更激进的优化

## 4. 关键编程要点


### 边界处理
```python
# 使用T.min确保不越界
offset = cid * block_size
tail_size = T.min(block_size, N - offset)
T.copy(A[offset], A_ub, [tail_size])
```
- `T.copy`自动对越界部分填零

### 数据格式转换
- **ND格式**: 标准内存布局
- **NZ格式**: NPU优化布局（用于矩阵运算）
- **ND→NZ**: `T.npuir_load_nd2nz()` - 加载到L1
- **NZ→ND**: `T.npuir_store_fixpipe(..., enable_nz2nd=True)` - 存储回GM

### 流水线同步
```python
# Cube核心完成后通知Vector核心
with T.Scope("Cube"):
    T.npuir_dot(...)
    T.sync_block_set(0)

# Vector核心等待Cube核心
with T.Scope("Vector"):
    T.sync_block_wait(0)
    T.npuir_add(...)
```

### 广播操作
支持自动广播：
```python
# 支持的广播模式
T.npuir_add(x_ub, y_ub, z_ub)  # [M,N] + [M,1]
T.npuir_add(x_ub, y_ub, z_ub)  # [1,N] + [1,1]

# 标量运算（必须先定义变量）
scalar_value = 2.5
T.npuir_mul(x_ub, scalar_value, x_ub)  # buffer * scalar
```

**重要**: 标量操作数必须在`T.Kernel`作用域内、`T.Scope`作用域外先定义为变量，不能直接使用字面量：
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    # 先定义标量变量
    scale_factor = 2.5
    offset_value = 1.5
    
    with T.Scope("Vector"):
        T.npuir_mul(x_ub, scale_factor, y_ub)  # 正确
        T.npuir_add(y_ub, offset_value, y_ub)  # 正确
        # T.npuir_mul(x_ub, 2.5, y_ub)  # 错误：不能直接用字面量
```

### 编译与运行
```python
import tilelang

# 编译为NPUIR（main是上面定义的@T.prim_func函数）
compiled = tilelang.compile(main, target="npuir")

# 运行（传入PyTorch NPU tensors）
compiled(A_tensor, B_tensor, C_tensor, N)
```

## 4. 常见编程模式详解

### 4.1 循环与索引计算

**串行循环 (T.serial)**:
```python
# K维度分块循环
for k_idx in T.serial(T.ceildiv(K, tile_k)):
    k_offset = k_idx * tile_k
    tail_k = T.min(tile_k, K - k_offset)
    
    # 加载当前tile
    T.npuir_load_nd2nz(A[m_offset, k_offset], A_l1, [tile_m, tail_k])
```

**二维块分解**:
```python
# 计算2D网格的块索引
n_blocks_m = T.ceildiv(M, tile_m)
n_blocks_n = T.ceildiv(N, tile_n)
total_blocks = n_blocks_m * n_blocks_n

with T.Kernel(total_blocks, is_npu=True) as (cid, _):
    # 从1D索引恢复2D坐标
    block_m_id = cid // n_blocks_n
    block_n_id = cid % n_blocks_n
    
    m_offset = block_m_id * tile_m
    n_offset = block_n_id * tile_n
```

**嵌套循环处理**:
```python
# 外层：M维度并行
with T.Kernel(T.ceildiv(M, tile_m), is_npu=True) as (m_id, _):
    # 中层：N维度串行
    for n_id in T.serial(T.ceildiv(N, tile_n)):
        # 内层：K维度累加
        for k_id in T.serial(T.ceildiv(K, tile_k)):
            # 矩阵乘法
            init = (k_id == 0)
            T.npuir_dot(A_l1, B_l1, C_l0c, initC=init)
```

### 4.2 Buffer索引与切片

**标量索引**:
```python
# 单元素访问
X[row_id, col_id]  # 访问(row_id, col_id)位置
Y[i]               # 访问1D tensor的第i个元素
```

**起始位置索引（用于T.copy）**:
```python
# 从offset位置开始拷贝tail_size个元素
T.copy(A[offset], A_ub, [tail_size])

# 2D拷贝：从(row, col)位置开始
T.copy(X[row_id, 0], x_ub[0, 0], [1, N])
```

**Buffer内部索引**:
```python
# 2D buffer的切片访问
x_ub[0, :]        # 第一行（所有列）
x_ub[:, 0]        # 第一列（所有行）
x_ub[0, k_offset] # 起始位置为(0, k_offset)
```

**注意事项**:
- 不支持: `A[start:end]` (Python切片语法)
- 支持: `A[offset]` + size参数指定范围
- 支持: `A_ub[0, :]` (buffer内部切片用于NPUIR操作)

### 4.3 数据拷贝模式

**GM ↔ UB (向量操作)**:
```python
# 标准拷贝
T.copy(A[offset], A_ub, [size])
T.copy(A_ub, A[offset], [size])

# 2D拷贝
T.copy(X[row, 0], x_ub[0, 0], [1, N])
```

**GM → L1 → L0 (矩阵运算)**:
```python
# GM到L1（ND→NZ格式）
T.npuir_load_nd2nz(A[m_offset, k_offset], A_l1, [tile_m, tile_k])

# L1到L0C（通过npuir_dot自动完成）
T.npuir_dot(A_l1, B_l1, C_l0c, initC=True)

# L0C到GM（NZ→ND格式）
with T.rs("PIPE_FIX"):
    T.npuir_store_fixpipe(
        C_l0c, C[m_offset, n_offset],
        size=[tile_m, tile_n],
        enable_nz2nd=True
    )
    T.sync_block_set(0)
```

### 4.4 边界处理技巧

**通用边界计算模式**:
```python
# 方法1: 两步计算（官方推荐）
offset = block_id * block_size
remaining = N - offset
tail_size = T.min(block_size, remaining)

# 方法2: 直接计算
tail_size = T.min(block_size, N - block_id * block_size)
```

**多维边界处理**:
```python
# 计算各维度的实际大小
m_offset = block_m * tile_m
n_offset = block_n * tile_n
k_offset = k_idx * tile_k

tail_m = T.min(tile_m, M - m_offset)
tail_n = T.min(tile_n, N - n_offset)
tail_k = T.min(tile_k, K - k_offset)

# 使用size参数传递实际大小
T.npuir_dot(A_l1, B_l1, C_l0c, size=[tail_m, tail_k, tail_n])
```

### 4.5 常用辅助函数

**向上取整除法**:
```python
n_blocks = T.ceildiv(N, block_size)  # 等价于 (N + block_size - 1) // block_size
```

**最小值函数**:
```python
actual_size = T.min(expected_size, available_size)
```

**无穷大常量**:
```python
neg_inf = -T.infinity("float32")  # 用于初始化max操作
pos_inf = T.infinity("float32")   # 用于初始化min操作
```

## 5. 性能优化要点

### Tile大小选择
- **对齐要求**: 8的倍数（默认），必须是2的幂次方
- **矩阵乘法**: M/N维度通常16-128, K维度通常64-256
- **向量操作**: 128-512之间

### 内存层次优化
1. 频繁访问数据放L1或L0
2. 中间结果保留在L0C
3. 仅在必要时写回GM

### 双核心协作
- Cube核心: 矩阵乘法、卷积
- Vector核心: 激活函数、归约、逐元素操作
- 使用同步原语协调

## 6. 高级编程主题

### 6.1 类型转换与精度管理

**不同精度计算**:
```python
# 输入float16，累加float32
A: T.Tensor((M, K), "float16")
C: T.Tensor((M, N), "float32")

# 类型转换
T.npuir_cast(fp16_ub, fp32_ub, "round")
```

**常见精度组合**:
- 矩阵乘法: 输入`float16`，累加器`float32`
- 向量运算: 统一使用`float32`
- Softmax/LayerNorm: 全程`float32`（数值稳定性）

### 6.2 多任务并行处理

**任务分发模式**:
```python
num_tasks = batch * seq_len
num_kernels = 32  # 并行核心数

with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
    # 每个核心处理多个任务
    for task_id in T.serial(T.ceildiv(num_tasks, num_kernels)):
        actual_task_id = task_id * num_kernels + kernel_id
        
        if actual_task_id < num_tasks:
            # 解析任务索引
            batch_id = actual_task_id // seq_len
            seq_id = actual_task_id % seq_len
            
            # 处理任务...
```

### 6.3 Workspace使用模式

**临时缓冲区**:
```python
# 函数签名中声明workspace
def main(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
    workspace: T.Tensor((M, N), dtype),  # 临时缓冲区
):
    # 使用workspace存储中间结果
    T.copy(temp_ub, workspace[offset], [size])
```

### 6.4 条件执行与控制流

**边界条件判断**:
```python
# 任务ID有效性检查
if logic_kernel_id < num_logic_kernels:
    # 执行任务
    ...

# K维度首次初始化
init_flag = (k_idx == 0)
T.npuir_dot(A_l1, B_l1, C_l0c, initC=init_flag)
```

**分支处理**:
```python
# 根据subid处理不同的子块
if subid == 0:
    # 处理前半部分
    offset = 0
else:
    # 处理后半部分
    offset = block_size // 2
```

### 6.5 数值稳定性技巧

**Softmax稳定计算**:
```python
# 1. 找到最大值
T.npuir_reduce(x_ub, max_ub, dims=[1], reduce_mode="max")

# 2. 减去最大值（防止溢出）
T.npuir_sub(x_ub, max_ub, x_ub)  # 自动广播 [1,N] - [1,1]

# 3. exp
T.npuir_exp(x_ub, x_ub)

# 4. 求和
T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")

# 5. 归一化
T.npuir_div(x_ub, sum_ub, x_ub)  # 自动广播 [1,N] / [1,1]
```

**小常量处理**:
```python
# 避免：科学计数法
eps = 1e-5  # 编译错误

# 方法1：完整形式，但避免小于0.0001
eps_value = 0.0001

# 方法2：tensor传递（推荐）
eps_tensor: T.Tensor((1, 1), "float32")  # 作为输入
```

## 7. 调试与验证

### 编译检查要点
```python
# 1. 函数名检查
@T.prim_func
def main(...):  # 必须是main
    pass

# 2. Shape格式检查
A: T.Tensor((M, N), dtype)  # 正确：元组
A: T.Tensor([M, N], dtype)  # 错误：列表

# 3. Buffer维度检查
x_ub = T.alloc_ub([1, N], dtype)  # 2D
sum_ub = T.alloc_ub([1, 1], dtype)  # 2D
```

### 常见运行时错误

**维度不匹配**:
```python
# 错误：1D buffer与2D buffer运算
x_1d = T.alloc_ub([N], dtype)        # 1D
y_2d = T.alloc_ub([1, N], dtype)     # 2D
T.npuir_add(x_1d, y_2d, ...)  # 维度不匹配错误

# 正确：统一维度
x_ub = T.alloc_ub([1, N], dtype)     # 2D
y_ub = T.alloc_ub([1, N], dtype)     # 2D
T.npuir_add(x_ub, y_ub, ...)  # 正确
```

**浮点常量错误**:
```python
# 错误：整数形式浮点数
scale = 5.0
T.npuir_mul(x_ub, scale, ...)  # 编译错误

# 正确：非整数浮点数或tensor
scale = 5.5  # 方法1
scale_tensor = ...  # 方法2（推荐）
```


**内核调用约定（out_idx）**

TileLang 的 `@tilelang.jit` / `tilelang.compile` 可以通过 `out_idx` 指定哪些张量属于输出。一旦设置了 `out_idx`，运行内核时只需要传入 `prim_func` 中定义的输入张量，输出会由 TileLang 自动创建并按顺序返回——这也是官方测试（如 `parallel_elementwise_static`）的默认写法。

```python
@tilelang.jit(out_idx=[1])
def parallel_elementwise_static(length=256):
    @T.prim_func
    def main(A: T.Tensor((length,), "float32"),
             B: T.Tensor((length,), "float32")):
        with T.Kernel(1, threads=length) as _:
            for i in T.Parallel(length):
                B[i] = A[i] + 1.0
    return main

kernel = parallel_elementwise_static()
result = kernel(data)              # ✅ 只传输入 data；TileLang 根据 out_idx 返回输出
```

常见错误是把 TileLang 内核当成 CUDA C 来写，例如先 `torch.empty_like` 一个输出张量，再调用 `kernel(x, y)`。由于 `prim_func` 只声明了两个输入（`A` / `B`），而你传了输入 `x` + 预分配的 `y` + out_idx 自动输出 `y`，运行时就会报 `ValueError: Expected 2 inputs, got 3 with 2 inputs and 1 outputs`。

> **实践建议**
>
> 1. **推荐方式**：保留 `out_idx`，调用时只传输入；返回值用 Python 解包即可，支持多输出（`out1, out2 = kernel(x, y)`）。
> 2. **确需手动管理输出**：不要设置 `out_idx`，并在 `prim_func` 里把输出张量也声明为输入参数，保证“定义多少参数就传多少参数”。
> 3. **提示/示例中强调**：在 KernelBench tilelang_cuda 任务的 prompt、文档和示例里明确提醒“使用 out_idx 时不要额外传输出张量”，可以减少 AIGC 生成代码时的参数数量错误。

