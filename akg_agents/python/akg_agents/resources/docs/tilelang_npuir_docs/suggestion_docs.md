# TileLang-Ascend 优化建议与常见错误

本文档提供TileLang-Ascend编程的关键规则、常见错误和性能优化建议。

---

## 关键规则速查

### 必须遵守的规则
1. **函数名**: `@T.prim_func def main(...)`
2. **Tensor形状**: 使用元组 `T.Tensor((M, N), dtype)`
3. **浮点常量**: 避免整数形式（`5.5`而非`5.0`）
4. **Buffer维度**: 所有参与运算的buffer必须相同rank
5. **科学计数法**: 不支持，使用Tensor传递或`0.0001`
6. **标量操作数**: 必须在`T.Kernel`内、`T.Scope`外预先定义为变量
7. **编译时常量**: 直接使用外部常量，不要在`@T.prim_func`内重新赋值
8. **复合表达式**: 必须拆分为多个步骤，不能直接作为函数参数（如`T.min(a, b-c)`需拆分）


---

## 1. 常见错误

### 错误1: 函数名不是main
```python
# 错误
@T.prim_func
def kernel_copy(...):
    ...

# 正确
@T.prim_func
def main(...):
    ...
```
**错误信息**: `rtFunctionRegister failed`

### 错误2: Tensor形状使用列表
```python
# 错误
T.Tensor([M, N], dtype)

# 正确
T.Tensor((M, N), dtype)
```
**错误信息**: `rtFunctionRegister failed`

### 错误3: 未设置is_npu=True
```python
# 错误
with T.Kernel(n_blocks) as (cid, _):

# 正确
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
```
**错误信息**: 编译可能通过，但运行时可能出现设备分配错误或功能异常

### 错误4: Buffer维度不一致
```python
# 错误
x_ub = T.alloc_ub([1, N], dtype)  # 2D
gamma_ub = T.alloc_ub([N], dtype)  # 1D
T.npuir_mul(x_ub, gamma_ub, x_ub)

# 正确：统一为2D
x_ub = T.alloc_ub([1, N], dtype)
gamma_ub = T.alloc_ub([1, N], dtype)
T.npuir_mul(x_ub, gamma_ub, x_ub)
```
**错误信息**: `IndexError: indexing 1 on an array of size 1`


### 错误6: 整数形式的浮点常量
```python
# 错误
value = 5.0
T.npuir_brc(value, buffer)

# 方案1: 使用非整数形式（快速修复）
value = 5.5  # 或 5.00001
T.npuir_brc(value, buffer)

# 方案2: 在buffer内定义（推荐）
# 需要初始化为5.0的场景
init_buffer: T.Tensor((N,), dtype)
# 在host端初始化为5.0后传入
T.npuir_brc(init_buffer[0], target_buffer)
```
**错误信息**: `unexpected decimal integer literal`

**根本原因**: 
- 浮点数 `5.0` 在编译器中会自动被转换为整型 `5`
- 后续的浮点运算不支持整型操作数,导致类型不匹配错误
- 通过添加小数部分(如 `5.00001`)可以防止转换为整型
- 或者直接在buffer中定义初始值,避免使用浮点字面量

### 错误7: 使用科学计数法
```python
# 错误
eps = 1e-5
eps = 0.00001  # 也会被自动转换为 1e-05

# 方案1: Tensor传递（推荐）
eps_tensor: T.Tensor((1, 1), dtype)
# 在host端初始化为1e-5后传入
T.npuir_add(x_ub, eps_tensor[0, 0], x_ub)

# 方案2: 使用较大值（如果精度允许）
eps = 0.0001  # 0.0001不会被转换

# 方案3: 在buffer内定义（适用于特殊值）
# 需要epsilon=1e-5的场景
eps_buffer: T.Tensor((1,), dtype)
# 在host端初始化为1e-5后传入
T.npuir_add(x_ub, eps_buffer[0], x_ub)

# 需要0.0的场景
zero_buffer: T.Tensor((N,), dtype)
# 在host端初始化为全0后传入
T.npuir_copy(zero_buffer, target_buffer)
```
**错误信息**: `custom op 'e' is unknown`

**根本原因**:
- 任何形式的 `0.00001` 大小的浮点数会被编译器自动转换为科学计数法 `1e-05`
- `0.0001` 不会被转换,但 `0.00001` 会被转换
- 后续编译阶段无法识别字符 `e`,将其误认为是一个未知的自定义操作符
- 通过Tensor传递或在buffer中预定义可以完全避免浮点字面量的转换问题

### 错误8: 标量操作数直接使用字面量
```python
# 错误
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    with T.Scope("Vector"):
        T.npuir_mul(x_ub, 2.5, y_ub)  # 直接使用字面量

# 正确
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    # 在T.Kernel内、T.Scope外定义变量
    scale_factor = 2.5
    
    with T.Scope("Vector"):
        T.npuir_mul(x_ub, scale_factor, y_ub)  # 使用变量
```
**错误信息**: `'float' object has no attribute 'buffer'`
**原因**: NPUIR编译器无法直接解析字面量标量，需要先定义为变量

### 错误9: T.copy维度不匹配
```python
# 错误
x_ub = T.alloc_ub([N], dtype)  # 1D
T.copy(X[row, 0], x_ub[0, 0], [1, N])  # 2D拷贝到1D buffer

# 正确
x_ub = T.alloc_ub([1, N], dtype)  # 2D
T.copy(X[row, 0], x_ub[0, 0], [1, N])  # 2D拷贝到2D buffer
```
**错误信息**: `'hivm.hir.load' op src and dst should have the same dimensions`

### 错误10: Grid Size计算位置错误（关键错误！）
```python
# ❌ 错误版本1：在@T.prim_func内部计算grid size
def create_kernel(N, block_size, dtype="float32"):
    @T.prim_func
    def main(X: T.Tensor((N), dtype), ..., N_param: T.int32):
        # ❌ 在prim_func内部计算 - 这是TVM表达式，不是Python值！
        n_blocks = T.ceildiv(N_param, block_size)
        
        with T.Kernel(n_blocks, is_npu=True) as (cid, _):  # ❌ 编译器无法解析
            ...

# ❌ 错误版本2：手动类型转换
def create_kernel(N, block_size, dtype="float32"):
    n_blocks = (N + block_size - 1) // block_size
    
    @T.prim_func
    def main(X: T.Tensor((N), dtype), ...):
        block_size_int = T.int32(block_size)  # ❌ 导致MLIR变量命名错误
        n_blocks = T.ceildiv(N, block_size_int)  # ❌ 重新计算
        
        with T.Kernel(n_blocks, is_npu=True) as (cid, _):  # ❌ 使用TVM表达式
            ...

# ✅ 正确版本：在闭包外部用Python计算（编译时确定）
def create_kernel(N, block_N, dtype="float32"):
    n_num = N // block_N  # ✅ Python整数除法，编译时确定
    
    @T.prim_func
    def main(X: T.Tensor((N), dtype), ..., shape: T.int32):
        # ✅ 不需要重新计算grid size
        
        with T.Kernel(n_num, is_npu=True) as (cid, _):  # ✅ 直接使用外部Python值
            # ✅ 内部只做运行时边界检查
            t0 = cid * block_N
            t0 = shape - t0
            tail_size = T.min(block_N, t0)
            ...
```
**错误信息**: `AttributeError: 'NoneType' object has no attribute 'group'`

**根本原因**: 
- `T.Kernel()`的第一个参数必须是**编译时可确定的Python整数**
- 如果在`@T.prim_func`内用`T.ceildiv()`计算，那是**TVM表达式**，不是Python值
- 编译器的`_parse_grid()`函数需要找到静态的`T.launch_thread("blockIdx.x", <static_number>)`模式
- 使用TVM表达式会导致match为None，无法注册kernel到NPU runtime

**正确模式**:
1. **编译时**: 在外部用Python计算grid size → `n_num = N // block_N`
2. **运行时**: 在内部用`shape`参数做边界检查 → `tail_size = T.min(block_N, shape - offset)`
3. **完美分离**: 编译时确定并行度，运行时处理边界

### 错误11: T.copy在T.Scope外部调用
```python
# ❌ 错误：数据传输操作在Scope外部
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    x_ub = T.alloc_ub([block_size], dtype)
    
    T.copy(X[offset], x_ub, [size])  # ❌ 在Scope外
    
    with T.Scope("Vector"):
        T.npuir_add(x_ub, y_ub, z_ub)

# ✅ 正确：所有数据操作在Scope内部
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    x_ub = T.alloc_ub([block_size], dtype)  # 分配可以在外部
    
    with T.Scope("Vector"):
        T.copy(X[offset], x_ub, [size])  # ✅ 在Scope内
        T.npuir_add(x_ub, y_ub, z_ub)
        T.copy(z_ub, Z[offset], [size])
```
**错误信息**: `TVMError: x_ub should be a memref`

**原因**: `T.copy`和所有`T.npuir_*`操作必须在`T.Scope("Vector")`或`T.Scope("Cube")`内部执行

**作用域规则**:
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    # ✅ 可以在这里: 分配buffer, 计算索引, 定义标量常量
    x_ub = T.alloc_ub([size], dtype)
    offset = cid * block_size
    scalar = 2.5
    
    with T.Scope("Vector"):  # 或 "Cube"
        # ✅ 必须在这里: T.copy, T.npuir_*
        T.copy(X[offset], x_ub, [size])
        T.npuir_mul(x_ub, scalar, y_ub)
        T.copy(y_ub, Y[offset], [size])
```

### 错误12: 编译时常量重新赋值
```python
# ❌ 错误：重新赋值外部常量
n_blocks_n = 128  # 外部定义

@T.prim_func
def main(...):
    with T.Kernel(total_blocks, is_npu=True) as (cid, _):
        n_blocks_n_int = n_blocks_n  # ❌ 重新赋值
        block_m_id = cid // n_blocks_n_int

# ✅ 正确：直接使用
@T.prim_func
def main(...):
    with T.Kernel(total_blocks, is_npu=True) as (cid, _):
        block_m_id = cid // n_blocks_n  # ✅ 直接使用外部常量
```
**错误信息**: `error: custom op 'v_' is unknown (tried 'func.v_' as well)`

**根本原因**: 在`@T.prim_func`内重新赋值外部编译时常量会导致变量被编译为未定义符号`v_`

**关键规则**: 
- ✅ 直接使用：`cid // n_blocks_n`
- ❌ 重新赋值：`temp = n_blocks_n; cid // temp`

### 错误13: 复合表达式未拆分
```python
# ❌ 错误：T.min参数中使用复合表达式
for k_idx in T.serial(T.ceildiv(K, block_k)):
    k_offset = k_idx * block_k
    tail_k = T.min(block_k, K - k_offset)  # ❌ 复合表达式

# ✅ 正确：拆分为两步
for k_idx in T.serial(T.ceildiv(K, block_k)):
    k_offset = k_idx * block_k
    k_remaining = K - k_offset  # ✅ 先计算
    tail_k = T.min(block_k, k_remaining)  # ✅ 再使用
```
**错误信息**: `error: expected SSA operand`

**根本原因**: TileLang无法将复合表达式直接降低到NPUIR，生成未解析的占位符

**关键规则**: 
- ✅ 拆分：`temp = K - k_offset; T.min(block_k, temp)`
- ❌ 直接使用：`T.min(block_k, K - k_offset)`
- 适用于`T.min/max`等所有内置函数的复合表达式参数


---

## 2. 性能优化

### 内存层次
优先级（速度）: L0 > L1 > UB > GM

```python
with T.Scope("Cube"):
    T.npuir_load_nd2nz(A_gm, A_l1, [m, k])
    T.npuir_dot(A_l1, B_l1, C_l0c)
    T.npuir_store_fixpipe(C_l0c, C_gm, ...)
```

### Tile大小
- 必须是2的幂次方
- 向量操作: 128-512
- 矩阵乘法: M/N维度16-128, K维度64-256

### 双核心协作
```python
with T.Scope("Cube"):
    T.npuir_dot(...)
    T.sync_block_set(0)

with T.Scope("Vector"):
    T.sync_block_wait(0)
    T.npuir_add(...)
```

---

## 3. 调试清单

### 基本检查
- [ ] 函数名为`main`
- [ ] Tensor用元组`(M, N)`
- [ ] 设置`is_npu=True`
- [ ] 指定`target="npuir"`

### Buffer检查
- [ ] 所有buffer相同rank
- [ ] 输入张量shape与buffer rank匹配
- [ ] `T.copy`源和目标rank一致
- [ ] Tile大小是2的幂次方

### 广播检查
- [ ] 编译器支持自动广播，直接使用即可
- [ ] 支持 `[M,N] op [M,1]` 和 `[1,N] op [1,1]`
- [ ] 支持 `buffer op scalar`（标量必须预先定义为变量）

### 常量检查
- [ ] 标量操作数在`T.Kernel`内、`T.Scope`外预先定义为变量
- [ ] 避免科学计数法
- [ ] 浮点常量使用非整数值
- [ ] 小常量（<0.0001）通过Tensor传递

---

## 4. 典型错误信息速查表

| 错误信息 | 原因 | 解决方案 | 错误编号 |
|---------|------|----------|---------|
| `rtFunctionRegister failed` | Tensor用列表或函数名不是main | 用元组、函数名改为main | 错误1/2 |
| `unexpected decimal integer literal` | 整数形式浮点数 | 使用`5.5`而非`5.0` | 错误6 |
| `custom op 'e' is unknown` | 科学计数法（如1e-5） | Tensor传递或`0.0001` | 错误7 |
| `custom op 'v_' is unknown` | 编译时常量重新赋值 | 直接使用外部常量，不要重新赋值 | 错误12 |
| `'float' object has no attribute 'buffer'` | 标量字面量直接用于运算 | 先定义变量再使用 | 错误8 |
| `IndexError: indexing 1 on an array of size 1` | 2D与1D buffer混用 | 统一所有buffer为相同rank | 错误4 |
| `'hivm.hir.load' op src and dst should have same dimensions` | `T.copy`维度不匹配 | 确保源和目标rank一致 | 错误9 |
| `'NoneType' object has no attribute 'group'` | block_size在T.Kernel内定义 | 在T.Kernel外定义常量 | 错误10 |
| `TVMError: x_ub should be a memref` | T.copy在T.Scope外调用 | 所有数据操作放在Scope内 | 错误11 |
| `expected SSA operand` | 复合表达式未拆分（如`K - k_offset`） | 将复合表达式拆分为多个步骤 | 错误13 |

---

## 5. 开发建议

### Buffer设计原则
1. 确定主要数据维度（如2D: `[M, N]`）
2. 统一所有buffer为相同rank
3. 输入张量shape与buffer rank一致

```python
# 推荐：统一2D
x_ub = T.alloc_ub([1, N], dtype)
gamma_ub = T.alloc_ub([1, N], dtype)
mean_ub = T.alloc_ub([1, 1], dtype)
```

### 常量处理
- **标量常量必须在`T.Kernel`内、`T.Scope`外定义为变量**
- 避免科学计数法
- 小常量通过Tensor传递

**示例**:
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
    # 在这里定义所有标量常量
    scale = 2.5
    offset = 1.5
    threshold = 0.5
    
    with T.Scope("Vector"):
        # 在这里使用变量
        T.npuir_mul(x_ub, scale, y_ub)
        T.npuir_add(y_ub, offset, y_ub)
        T.npuir_max(y_ub, threshold, y_ub)
```

### 渐进式开发
1. 先实现基础版本
2. 验证正确性
3. 逐步添加优化
4. 性能测试

---

## 附录: 完整示例

### Softmax 示例
```python
@T.prim_func
def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype), shape: T.int32):
    with T.Kernel(M, is_npu=True) as (row_id, _):
        x_ub = T.alloc_ub([1, N], dtype)
        max_ub = T.alloc_ub([1, 1], dtype)
        sum_ub = T.alloc_ub([1, 1], dtype)
        
        with T.Scope("Vector"):
            T.copy(X[row_id, 0], x_ub[0, 0], [1, N])
            
            # 找最大值，数值稳定
            T.npuir_reduce(x_ub, max_ub, dims=[1], reduce_mode="max")
            T.npuir_sub(x_ub, max_ub, x_ub)  # [1,N] - [1,1] 自动广播
            
            # 指数和归一化
            T.npuir_exp(x_ub, x_ub)
            T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")
            T.npuir_div(x_ub, sum_ub, x_ub)  # [1,N] / [1,1] 自动广播
            
            T.copy(x_ub[0, 0], Y[row_id, 0], [1, N])
```

### 科学计数法处理
```python
# Tensor传递方式
@T.prim_func
def main(X: T.Tensor((M, N), dtype), 
         eps_tensor: T.Tensor((1, 1), dtype),
         Y: T.Tensor((M, N), dtype)):
    with T.Kernel(M, is_npu=True) as (row_id, _):
        eps_ub = T.alloc_ub([1, 1], dtype)
        var_ub = T.alloc_ub([1, 1], dtype)
        
        with T.Scope("Vector"):
            T.copy(eps_tensor[0, 0], eps_ub[0, 0], [1, 1])
            T.npuir_add(var_ub, eps_ub, var_ub)

# Python侧
eps_value = 1e-5
eps_tensor = torch.full((1, 1), eps_value, dtype=torch.float32).npu()
compiled_kernel(X, eps_tensor, Y)
```

