---
name: tilelang-ascend-error-fix
description: "TileLang-Ascend 算子编码常见错误与修复指南。涵盖编译时错误（内存分配失败、维度不匹配、API参数错误、GEMM除零、Autotune问题）、运行时错误（结果不正确、精度问题）。"
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang-Ascend 算子编码常见错误与修复

---

## 编译时错误

### 1. 内存分配失败

**错误信息**:
```
TVMError: Memory allocation failed for: buffer_name required: XXXX, new memory available: YYYY
```

**原因**: UB空间不足，所有buffer总大小超过限制

**解决方案**:
1. 减小分块大小：
   ```python
   block_M, block_N = 64, 128
   ```
2. 开启自动内存规划以复用buffer：
   ```python
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   ```
3. 减少中间buffer数量，尽可能复用

### 2. 维度不匹配

**错误信息**:
```
error: Source and Dest dimension must match.
```

**原因**: broadcast操作的源和目标shape不符合要求

**解决方案**:
确保源buffer的shape为 `[M, 1]` 或 `[1, N]`，目标buffer为 `[M, N]`：

```python
max_ub = T.alloc_ub([block_M // VEC_NUM, 1], dtype)
max_2d_ub = T.alloc_ub([block_M // VEC_NUM, block_N], dtype)
T.tile.broadcast(max_2d_ub, max_ub)
```

### 3. API参数错误

**错误信息**:
```
error: max() takes 3 positional arguments but 4 were given
```

**原因**: API调用参数不正确

**解决方案**:
查看 API 指南确认正确的参数签名：

```python
T.tile.max(dst, src0, src1)
```

### 4. GEMM 除零编译错误

**错误信息**:
```
InternalError: Check failed: pb->value != 0 (0 vs. 0) : Divide by zero
 --> ...py:65:18  bx = cid // n_num
```

**原因**: `n_num = N // block_N = 0`（当 `block_N > N`），导致 `cid // 0`。

**解决方案**: 在调用 GEMM 前确保 M, N ≥ block size。如果 `M < block_M` 或 `N < block_N`，zero-padding 矩阵到 block 倍数再调用 GEMM，完成后裁剪。

```python
M_pad = ((M + block_M - 1) // block_M) * block_M
N_pad = ((N + block_N - 1) // block_N) * block_N
K_pad = ((K + block_K - 1) // block_K) * block_K

if M_pad > M or K_pad > K:
    kernel_padded = torch.zeros(M_pad, K_pad, ...)
    kernel_padded[:M, :K] = kernel_flat

output = output[:M, :N]
```

**关键约束**: 不 padding 时 `M // block_M = 0`（当 M < block_M）会导致零 block 启动（输出全零）或除零编译崩溃。

### 5. Autotune supply_prog IndexError

**错误信息**:
```
An error occurred while testing config {...}
```

**原因**: `supply_prog(params)` 中 `params` 仅含输入 tensor 描述符（不含输出），`params[2]` 访问越界。

**解决方案**: 从 `params[0].shape` 和 `params[1].shape` 提取维度：
```python
def supply_prog(params):
    M_val, K_val = int(params[0].shape[0]), int(params[0].shape[1])
    _, N_val = int(params[1].shape[0]), int(params[1].shape[1])
    return [torch.randn(M_val, K_val).half().npu(), torch.randn(K_val, N_val).half().npu()]
```

### 6. Autotune get_configs 参数格式错误

**错误信息**:
```
TypeError: get_configs() missing 1 required positional argument: 'K'
```

**原因**: autotuner 调用 `get_configs` 时传参为 `(key_args_tuple, key_kwargs_tuple)`，即 `((M,N,K), ())`。直接声明 `get_configs(M, N, K)` 会收到 tuple 而非 3 个 int。

**解决方案**: 签名为 `get_configs(key_args, _key_kwargs=None)`，从 `key_args` 解包 M, N, K。调用时传递 callable 引用（`configs=get_configs`），而非调用结果（`configs=get_configs()`）。

### 7. L0C 溢出 Segfault

**现象**: autotune 编译通过但 benchmark 时进程直接 crash（Segfault），无 Python 异常。

**可能原因之一**：当 `block_M * block_N * sizeof(accum_dtype) > L0C_capacity` 时，可能导致片上 buffer 使用超过硬件限制。例如 A2/A3 设备 L0C 为 128KB，float32 accum 元素数不应超过 32768。

**解决方案**: autotune 的 `get_configs` 中过滤超大 block：
```python
block_M = [bs for bs in [64, 128] if bs <= M]
```

**Autotune config 过滤规则**（必须在 `get_configs` 中执行）：
1. 过滤 `block > dimension` 的无效组合（避免除零编译错误）
2. 过滤 `block_M * block_N * sizeof(accum_dtype) > L0C_capacity` 的组合（避免 L0C 溢出 segfault）。A2/A3 设备 L0C = 128KB，float32 accum 时 `block_M * block_N ≤ 32768`

```python
def get_configs(key_args, _key_kwargs=None):
    M, N, K = key_args
    configs = []
    for bm in [64, 128]:
        for bn in [64, 128]:
            for bk in [32, 64]:
                if bm > M or bn > N or bk > K:
                    continue
                if bm * bn * 4 > 131072:  # L0C 128KB limit for float32 accum
                    continue
                configs.append({"block_M": bm, "block_N": bn, "block_K": bk})
    return configs
```

### 8. `InternalError: Duplicate buffer name found: tmp_ub`

**原因**：TileLang 编译器内部 pass 会自动创建临时 buffer。这些临时 buffer 可能有固定的名字（如 `tmp_ub`），如果自定义同名 buffer，就会产生冲突。

**常见错误代码**：
```python
@T.prim_func
def main(...):
    with T.Kernel(N, is_npu=True):
        tmp_ub = T.alloc_ub([block_size], "float32")  # ❌ 自定义的 tmp_ub
        T.tile.broadcast(tmp_2d, mean_ub)              # 编译器也会创建 tmp_ub
```

**解决方案**：避免使用以下名字作为用户定义的 buffer 名称：
- `tmp_ub`、`tmp`、`tmp_buf`
- `broadcast_workspace_*`
- 其他可能与编译器内部 pass 冲突的名字

---

## 运行时错误

### 1. 结果不正确

**可能原因**:
1. 缺少同步
2. 公式实现错误
3. 数据类型问题

**解决方案**:

1. 根据编程模式处理同步。Expert 手动模式需要显式同步；Developer / 混合模式开启 `TL_ASCEND_AUTO_SYNC` 后不要额外插入手动 barrier。
   ```python
   pass_configs = {
       tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
       tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
   }
   ```

2. 检查数据类型是否匹配

### 2. 精度问题

#### 输出与参考实现有微小差异

**原因**: float16精度较低，累积误差

**解决方案**: 使用float32进行累加计算

#### kernel 运行正常不报错，但输出全为 0。

**可能原因**:
1. 标量与向量的算术运算未用 T.tile API。标量与向量之间的算术运算不能用 `+ - * /` 运算符直接写，必须用 `T.tile` 系列 API。
2. kernel 编译时指定了 `out_idx=[...]`，但调用时丢弃了返回值，写成了 `kernel_func(*tensors)` 。而传入 out_idx 时，kernel 的输出只会通过返回值返回，不会原地写入，因此输出为预定义的空值。

**解决方案**: 
1. 改用 `T.tile` API 实现标量-向量运算
2. 若 kernel 编译时指定了 `out_idx=[...]`，调用时必须接收返回值：`outputs = kernel_func(*inputs)`。

#### 只有前大约 64（float32）/128（float16）个元素正确，后续全部错误，且 kernel 中使用了 `T.tile.select`

**原因**: `T.tile.select` 的 `selMode` 误设置成 `VSEL_CMPMASK_SPR`。AscendC Select 支持三种模式：

| 模式 | 枚举值 | mask 消耗方式 | 每迭代有效 bit | 适用场景 |
|------|--------|-------------|--------------|---------|
| VSEL_CMPMASK_SPR | 0 | 每迭代重复使用相同 mask | float32 前 64 bit（8 bytes） | mask 全相同或固定 |
| VSEL_TENSOR_SCALAR_MODE | 1 | 每迭代连续消耗 mask | 无限制 | tensor vs scalar 选择 |
| VSEL_TENSOR_TENSOR_MODE | 2 | 每迭代连续消耗 mask | 无限制 | tensor vs tensor 选择 |

`VSEL_CMPMASK_SPR` 每轮迭代都使用同一个固定的前 64 bit mask 处理所有数据——如果你的 mask 是 `compare(x > 0)` 的结果（不同位置 bit 不同），这个模式会导致元素 64+ 的选择完全错误。

**解决方案**: 将 `VSEL_CMPMASK_SPR` 替换为 `VSEL_TENSOR_SCALAR_MODE` 或 `VSEL_TENSOR_TENSOR_MODE`
