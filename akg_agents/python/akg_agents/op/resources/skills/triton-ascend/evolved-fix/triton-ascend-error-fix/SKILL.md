---
name: triton-ascend-error-fix
description: triton-ascend 常见错误修复：UB/CBUF溢出、BiShengIR编译失败、语法限制违反、数值正确性、多维索引分解错误、张量连续性
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
---

## 1. UB / CBUF 溢出

- **报错特征**: `cbuf overflow`
- **根因**: 分块参数（BLOCK_M/N/K）过大，超出 Ascend UB 容量
- **修复**: 减小分块参数，通常 BLOCK_M=64, BLOCK_N=128 是安全起点

```python
# 错误：分块过大导致 UB overflow
BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 256

# 修复：安全起始点
# CUBE (matmul fp16): BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
# CUBE (matmul fp32): BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
# VEC (elementwise):  BLOCK_SIZE=1024~2048
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
```

4D tensor 矩阵乘法中 batch 维度额外占用 UB 空间，建议将 batch 展开到 grid 维度而非内层循环。

## 2. BiShengIR / HiVM 编译失败

- **报错特征**: `hivm.hir.vsel: Unsupported op for finding the root alloc`、`Failed to run BiShengHIR pipeline`
- **根因**: 编译器后端不支持复杂的 mask 组合或指针运算模式

### 2a. 内联地址计算过于复杂

```python
# 错误
tl.store(c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn, acc, mask=mask)

# 修复：拆分为中间变量
c_ptrs = c_ptr + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
tl.store(c_ptrs, acc, mask=mask)
```

### 2b. `tl.where` + 复杂 mask 导致 vsel 错误

```python
# 错误：嵌套 mask + tl.where 触发 hivm.hir.vsel 错误
a_tri_mask = a_offsets_k[None, :] >= a_offsets_m[:, None]
a_valid_mask = a_mask_m & a_mask_k
a = tl.where(a_tri_mask & a_valid_mask, a, 0.0)

# 修复：改用乘法替代 tl.where（将 bool mask 转为 float 后相乘）
a_tri_mask = (a_offsets_k[None, :] >= a_offsets_m[:, None]).to(tl.float16)
a_valid_mask = (a_mask_m).to(tl.float16) * (a_mask_k).to(tl.float16)
a = a * a_tri_mask * a_valid_mask
```

## 3. Triton 语法限制违反

### 3a. 禁止 `continue` / `break` / `return`

```python
# 错误：unsupported AST node type: Continue
for i in range(N):
    if condition:
        continue
    do_work()

# 修复：用 if-else 包裹
for i in range(N):
    if not condition:
        do_work()
```

### 3b. constexpr 索引错误

```python
# 错误：ValueError('unsupported tensor index: constexpr[0]')
result = tl.sum(data, axis=0)
tl.atomic_add(out_ptr, result[0])

# 修复：tl.sum 已返回标量，直接使用
result = tl.sum(data, axis=0)
tl.atomic_add(out_ptr, result)
```

### 3c. tensor.cast 类型不兼容

```python
# 错误：cast incompatible shapes
result = tl.dot(a_fp16, b_fp16)

# 修复：显式指定 fp32 累加器
result = tl.dot(a_fp16, b_fp16, acc=tl.zeros([M, N], dtype=tl.float32))
```

### 3d. 禁止语法速查

| 禁止 | 替代 |
|------|------|
| `continue` / `break` / `return` | `if-else` 包裹 |
| `while` 循环 | `for` + `if` |
| `lambda` | 命名函数 |
| `a and b` / `a or b`（tensor） | `a & b` / `a \| b` |
| Python 切片 `tensor[1:3]` | `tl.arange` + mask |
| `tl.where(cond, ptr_a, ptr_b)` | if/else 静态分支 |
| `result[0]` 索引 constexpr | 直接使用标量 result |

## 4. 数值正确性问题

- **报错特征**: `AssertionError: 输出不一致, err_cnt=XXXX`

### 4a. 三角矩阵 mask 方向错误

```python
# 上三角: 确保 col >= row 的区域非零
row_idx = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
col_idx = block_k * BLOCK_K + tl.arange(0, BLOCK_K)
tri_mask = col_idx[None, :] >= row_idx[:, None]
a = tl.load(a_ptr + ..., mask=tri_mask & bounds_mask, other=0.0)
```

### 4b. 4D tensor 维度分解错误

```python
# 正确的 batch 维度分解
pid = tl.program_id(0)
batch_idx = pid // num_blocks_per_batch
block_idx = pid % num_blocks_per_batch
b0 = batch_idx // dim1
b1 = batch_idx % dim1
```

### 4c. Reduction 精度丢失

```python
# fp32 累加器避免 fp16 精度丢失
acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(...)  # fp16
    b = tl.load(...)  # fp16
    acc += tl.dot(a, b)  # fp32 累加
result = acc.to(tl.float16)
```

## 5. 多维循环索引分解错误

- **报错特征**: 计算结果不正确（无编译报错）
- **根因**: 将多维问题展平为一维后，索引分解/还原逻辑出错。常见于 Norm、Pooling、多 batch 算子
- **修复**: 使用清晰的循环嵌套 + 显式的维度分解，避免脆弱的一维展平映射

```python
# 易错：复杂的一维展平映射
total_tasks = N * G
for task_idx in range(pid, total_tasks, CORE_NUM):
    n_idx = task_idx // G
    g_idx = task_idx % G
    c_local = offsets // S       # 含义不清晰，容易写错
    hw = offsets - c_local * S

# 推荐：显式的多层嵌套 + 清晰的局部索引
for n in range(pid, N, CORE_NUM):
    for g_idx in range(num_groups):
        for i in range(0, group_elems, BLOCK_SIZE):
            local_idx = i + tl.arange(0, BLOCK_SIZE)
            c_local = local_idx // hw_size
            spatial_idx = local_idx % hw_size
```

## 6. 张量连续性

在 kernel wrapper 入口处强制 `.contiguous()`：

```python
if not x.is_contiguous():
    x = x.contiguous()
```

---
## Quick Checklist

1. **编译失败 + "ub overflow" / "cbuf overflow"** → 缩小 BLOCK 尺寸（§1）
2. **编译失败 + "hivm.hir" / "root alloc"** → 简化 mask / 拆分指针运算（§2）
3. **编译失败 + "unsupported AST"** → 检查禁止语法表（§3）
4. **验证失败 + "err_cnt"** → 检查 mask 方向、索引计算、精度（§4）
5. **结果不正确但无报错** → 检查多维索引分解逻辑（§5）
6. **结果 NaN 或静默错误** → 检查连续性（§6）
