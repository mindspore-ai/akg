---
name: tilelang-ascend-mode-guide
description: "TileLang Ascend Developer/Expert 模式选择与 pass_configs 配置指南。当需要确定编程模式、配置 pass_configs、或在两种模式之间转换时触发。API 详情请参考 tilelang-ascend-api skill。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
---

# TileLang Ascend 编程模式与 pass_configs 指南

 **API 用法详情**（内存分配、计算原语、同步原语等）请参考 TileLang Ascend API 最佳实践 章节。

---

## 1. 模式对比

| 维度 | Developer 模式 | Expert 模式 |
|------|---------------|-------------|
| **内存分配** | `T.alloc_shared` / `T.alloc_fragment` | `T.alloc_L1` / `T.alloc_ub` / `T.alloc_L0A/L0B/L0C` |
| **计算表达** | `T.Parallel` + 符号运算 | `T.tile.xxx` 扩展原语 |
| **作用域** | 编译器自动分离 Cube/Vector | 手动 `with T.Scope("C"/"V")` |
| **同步** | 编译器自动插入 | 手动 `T.barrier_all` / `T.set_flag` / `T.wait_flag` |
| **pass_configs** | **全部开启** | **全部关闭或不设** |
| **适用场景** | 大多数算子，跨平台兼容 | 极致性能优化，需要底层控制 |

**混合模式**：Developer 主体 + 少量 Expert / Ascend 专属 `T.tile.xxx`。使用 Developer 的 pass_configs，不写 `T.Scope` 和手动同步。大多数实际算子使用混合模式。

---

## 2. pass_configs 详解（核心）

### 2.1 四个 Ascend 专用开关

```python
import tilelang

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # ① 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,   # ② 自动内存规划
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,   # ③ 自动CV分离
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,      # ④ 自动核间同步
}
```

#### ① TL_ASCEND_AUTO_SYNC（自动核内同步）

- **底层 key**：`"tl.ascend_auto_sync"`，默认 False
- **功能**：自动在数据搬运和计算之间插入 `T.barrier_all()` 等同步指令
- **开启时**：无需手写 `T.barrier_all()`、`T.set_flag`/`T.wait_flag`
- **关闭时**：必须手动插入所有同步点

#### ② TL_ASCEND_MEMORY_PLANNING（自动内存规划）

- **底层 key**：`"tl.ascend_memory_planning"`，默认 False
- **功能**：自动分析 buffer 生命周期，实现片上内存复用
- **开启时**：自动复用 buffer 空间，减少片上内存占用
- **关闭时**：需手动通过 `T.annotate_address` 规划内存地址


#### ③ TL_ASCEND_AUTO_CV_COMBINE（自动 CV 分离）

- **底层 key**：`"tl.ascend_auto_cv_combine"`，默认 False
- **功能**：自动将 kernel 中的 Cube 操作和 Vector 操作分离到不同的执行核
- **开启时**：无需手写 `with T.Scope("C")` / `with T.Scope("V")`
- **关闭时**：必须手动用 `T.Scope` 标注每段代码的执行域

#### ④ TL_ASCEND_AUTO_CV_SYNC（自动核间同步）

- **底层 key**：`"tl.ascend_auto_cross_core_sync"`，默认 False
- **功能**：自动在 Cube Scope 和 Vector Scope 之间插入 `T.set_cross_flag`/`T.wait_cross_flag`
- **开启时**：无需手写核间同步
- **关闭时**：必须手动管理核间同步

### 2.2 按场景选择 pass_configs


| 场景 | AUTO_SYNC | MEMORY_PLANNING | AUTO_CV_COMBINE | AUTO_CV_SYNC |
|------|-----------|-----------------|-----------------|--------------|
| **纯 Vector 算子**（elementwise, softmax） | ✅ | ✅ | 不需要 | 不需要 |
| **Developer GEMM** | ✅ | ✅ | ✅ | ✅ |
| **Developer Flash Attention（核间流水线）** | ✅ | 视情况 | ✅ | ✅ |
| **Expert 极致性能** | ❌ | ❌ | ❌ | ❌ |
| **混合模式** | ✅ | ✅ | ✅ | ✅ |

**纯 Vector 算子**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

**Developer GEMM**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

**Expert 全手动**：
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}
```

---

## 3. 模式转换规则（Expert → Developer）

### 3.1 转换步骤

1. **开启 pass_configs**：添加完整 4 个 True 开关
2. **内存分配**：`T.alloc_L1` → `T.alloc_shared`，`T.alloc_L0C` → `T.alloc_fragment`，`T.alloc_ub` → `T.alloc_shared`
3. **删除作用域**：移除 `with T.Scope("C")` / `with T.Scope("V")`
4. **删除同步**：移除 `T.barrier_all()`、`T.set_flag`/`T.wait_flag`、`T.set_cross_flag`/`T.wait_cross_flag`
5. **计算转换**（可选）：`T.tile.exp(dst, src)` → `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])`
6. **删除手动内存规划**：移除 `T.annotate_address`


### 3.2 转换对照表

| Expert 写法 | Developer 写法 |
|-------------|---------------|
| `T.alloc_L1(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_ub(shape, dtype)` | `T.alloc_shared(shape, dtype)` |
| `T.alloc_L0A/L0B(shape, dtype)` | 删除（`gemm_v0` 内部处理） |
| `T.alloc_L0C(shape, dtype)` | `T.alloc_fragment(shape, dtype)` |
| `with T.Scope("C"): ...` | 直接写代码（编译器自动分离） |
| `T.barrier_all()` | 删除（编译器自动插入） |
| `T.set_flag/T.wait_flag(...)` | 删除 |
| `T.set_cross_flag/T.wait_cross_flag(...)` | 删除 |
| `T.tile.exp(dst, src)` | `for i,j in T.Parallel(...): dst[i,j] = T.exp(src[i,j])` 或保留 |
| `T.annotate_address({...})` | 删除（开启 MEMORY_PLANNING） |

---

## 4. Developer vs Expert 模式代码对比

---

### 4.1 Developer 模式

**特点**：
- 无 `T.Scope`、无 `T.barrier_all`、无 `T.set_flag`
- 使用 `alloc_shared` / `alloc_fragment`
- 全靠 pass_configs 自动处理同步和内存

---

### 4.2 Expert 模式

**特点**：
- 手动 `T.barrier_all()` 同步
- 使用 `alloc_L1` / `alloc_L0C` 显式指定存储层级
- 无 pass_configs

---

### 4.3 Expert 模式 pass_configs

Expert 模式极致性能场景，**全部关闭**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

### 4.4 Developer 核间流水线 pass_configs

核间流水线场景，**全部开启**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}

@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(...):
    ...
```

---

### 4.5 混合模式

混合模式典型场景：Developer pass_configs + Ascend 专属 `T.tile` 原语（`T.tile.fill/max/sub/exp/div`）

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

# kernel 内部混用 Developer 和 Expert API
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    # Expert API：T.tile.fill, T.tile.max, T.tile.sub, T.tile.exp 等
    T.tile.fill(acc_ub, 0.0)
    T.reduce_max(scores_ub, row_max_ub, dim=-1)
    T.tile.sub(scores_ub, scores_ub, row_max_ub)
    T.tile.exp(scores_ub, scores_ub)
    T.reduce_sum(scores_ub, row_sum_ub, dim=-1)
    T.tile.div(scores_ub, scores_ub, row_sum_ub)
    # 使用 Developer 的 pass_configs 自动处理同步
```

**关键点**：`T.tile.xxx` 和 `T.reduce_*` 可以在 Developer pass_configs 下正常工作，无需手写同步。
