# TileLang-Ascend API 速查

## 1. 内核定义

### @T.prim_func
```python
@T.prim_func
def main(A: T.Tensor((M, N), dtype), ...):
```
- 函数名必须为`main`，Tensor用元组`(M, N)`

### T.Kernel(n_blocks, is_npu=True)
```python
with T.Kernel(n_blocks, is_npu=True) as (cid, _):
```
- `n_blocks`必须是编译时Python整数
- `is_npu=True`必须设置

## 2. 内存分配

```python
A_ub = T.alloc_ub([256], "float32")      # UB: 向量操作
A_l1 = T.alloc_L1([16, 256], "float16")  # L1: 矩阵输入
C_l0c = T.alloc_L0C([16, 16], "float32") # L0C: 矩阵累加器
```

## 3. 数据传输

```python
# 向量数据拷贝
T.copy(A[offset], A_ub, [size])

# 矩阵加载 (ND→NZ格式)
T.npuir_load_nd2nz(A[m, k], A_l1, [tile_m, tile_k])

# 矩阵存储 (NZ→ND格式)
T.npuir_store_fixpipe(C_l0c, C[m, n], size=[tile_m, tile_n], enable_nz2nd=True)
```

## 4. 算术运算

```python
# 基本运算
T.npuir_add/sub/mul/div(A_ub, B_ub, C_ub)  # C = A op B
T.npuir_max/min(A_ub, B_ub, C_ub)          # C = max/min(A, B)
T.npuir_exp(x_ub, exp_ub)                   # exp(x)

# 标量运算（必须先定义变量）
scale = 2.5
T.npuir_mul(A_ub, scale, B_ub)  # buffer * scalar
```
- 支持广播: `[M,N] op [M,1]`, `[1,N] op [1,1]`, `buffer op scalar`
- 标量必须在`T.Kernel`内、`T.Scope`外定义

## 5. 矩阵运算

```python
# C = C + A × B
T.npuir_dot(A_l1, B_l1, C_l0c, initC=True)  # 首次初始化
T.npuir_dot(A_l1, B_l1, C_l0c, initC=False) # 后续累加
```
- `initC=True`: 初始化C为0
- `size=[m, k, n]`: 可选，手动指定矩阵大小

## 6. 归约与类型转换

```python
# 归约
T.npuir_reduce(x_ub, sum_ub, dims=[1], reduce_mode="sum")
# reduce_mode: "sum", "prod", "max", "min"

# 类型转换
T.npuir_cast(fp16_ub, fp32_ub, "round")
# round_mode: "round"/"floor"/"ceil"

# 广播
T.npuir_brc(scalar, buffer_ub)
```

## 7. 同步与控制

```python
# 核心Scope
with T.Scope("Cube"):    # 矩阵运算
with T.Scope("Vector"):  # 向量运算

# 流水线资源
with T.rs("PIPE_FIX"):   # L0C存储
    T.npuir_store_fixpipe(...)

# 同步
T.sync_block_set(0)
T.sync_block_wait(0)
```

## 8. 辅助函数

```python
n_blocks = T.ceildiv(N, block_size)  # 向上取整除法
tail_size = T.min(block_size, N - offset)  # 边界处理
for k in T.serial(n_iter):  # 串行循环
```

## 9. 重要约束

### 内核定义
- 函数名必须为`main`
- Tensor形状用元组 `(M, N)`
- 编译指定`target="npuir"`

### 内核启动
- 必须`is_npu=True`
- Grid size必须是编译时Python整数
- 只支持一维网格

### Buffer约束
- 参与运算的buffer必须相同rank
- 推荐统一用2D buffer `[M, N]`

### 常量约束
- 避免整数形式浮点数 (用`5.5`不用`5.0`)
- 不支持科学计数法 (用`0.0001`不用`1e-5`)
- 标量操作数必须在`T.Kernel`内、`T.Scope`外定义

### 数据类型
- `"float16"`, `"float32"`, `"int32"`
