---
name: triton-ascend-a5-attention
description: "适用于 A5（Ascend950）注意力(attention)机制算子的串行优化指南。当算子的核心计算是 Transformer 风格的注意力运算，且目标硬件为 Ascend950 时应选择此指南。涵盖 Cube/Vector 操作、al.fixpipe/bl.alloc 数据流、串行同步机制（sync_block_set/wait）、Flash Attention 四阶段分解、P 矩阵 ND→NZ 格式转换等 A5 专用技巧。本文档给出的是 Cube/Vector 串行交替执行版本，不适用于不含注意力结构的普通矩阵乘法或归约运算。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  note: "A5(Ascend950) Cube/Vector 亲和接口串行版"
  operator_type: "attention"
---

# A5 Flash Attention 串行优化指南

## 1. A5 硬件架构与 Flash Attention 映射

Ascend950 AI Core 包含两种关键计算单元，Flash Attention 的四个阶段映射到这两种 core 上：

| 单元 | 职责 | Flash Attention 中的角色 |
|------|------|--------------------------|
| **Cube 核** | 矩阵乘法 | QK matmul、PV matmul |
| **Vector 核** | 逐元素运算 | softmax、exp、归一化、flash_update |

## 2. 存储层级与数据流

```
GM (Q/K/V/Out)
  ↓ tl.load
Cube: Q@K^T → L0C
  ↓ al.fixpipe (NZ2ND, ROW_SPLIT)
UB: qk_ub (BLOCK_M//2, BLOCK_N)  ← ROW_SPLIT 拆给两个 sub-vector
  ↓ Vector: softmax → p_nz
  ↓ al.copy (UB → L1)
L1: p_l1 (NZ 分形格式)
  ↓ Cube: P@V → L0C
  ↓ al.fixpipe (NZ2ND, ROW_SPLIT)
UB: pv_ub (BLOCK_M//2, HEAD_DIM)
  ↓ Vector: flash_update → acc
  ↓ tl.store
GM: Out
```

关键点：
- `al.fixpipe` 将 Cube 的 L0C 结果搬到 UB，使用 `ROW_SPLIT` 模式自动拆给两个 sub-vector core
- `bl.alloc` 在 UB/L1 上分配片上 buffer，供 Cube 和 Vector 共享
- `bl.to_tensor` 在 Vector scope 中读取 fixpipe 写入的 UB 数据
- `al.copy` 将 P 矩阵从 UB 搬到 L1 供 Cube 做 PV matmul

## 3. 串行同步机制

串行模式下 Cube 和 Vector 在每次 N-loop 迭代中交替执行，使用 3 个同步事件：

```
Cube:  QK matmul → fixpipe → [set 0] → [wait 1] → PV matmul → fixpipe → [set 2]
Vector:              [wait 0] → softmax → copy P→L1 → [set 1] → [wait 2] → flash_update
```

| Event | 方向 | sender_pipe → receiver_pipe | 含义 |
|-------|------|-----|------|
| 0 | cube→vector | `PIPE_FIX` → `PIPE_V` | QK fixpipe 完成，qk_ub 就绪 |
| 1 | vector→cube | `PIPE_MTE3` → `PIPE_MTE1` | P 已 copy 到 L1，p_l1 就绪 |
| 2 | cube→vector | `PIPE_FIX` → `PIPE_V` | PV fixpipe 完成，pv_ub 就绪 |

## 4. Kernel 结构设计

### 4.1 Buffer 分配（kernel 入口）

```python
qk_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)
pv_ub = bl.alloc(tl.float32, (BLOCK_M // 2, HEAD_DIM), al.ascend_address_space.UB)
p_l1  = bl.alloc(cast_dtype, (BLOCK_N // 16, BLOCK_M // 16, 16, 16), al.ascend_address_space.L1)
```

- UB buffer 的行维度使用 `BLOCK_M // 2`，因为 ROW_SPLIT 模式下每个 sub-vector 只处理一半
- L1 buffer 使用 NZ 分形格式 `(BLOCK_N//16, BLOCK_M//16, 16, 16)`

### 4.2 Cube Scope

```python
with al.scope(core_mode="cube"):
    for start_n in range(0, N_Loop, 1):
        _qk_matmul(q, K_block_ptr, qk_ub, HEAD_DIM, BLOCK_N)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
        _pv_matmul(p_l1, pv_ub, V_block_ptr, HEAD_DIM, BLOCK_M, BLOCK_N)
        al.sync_block_set("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
```

### 4.3 Vector Scope

```python
with al.scope(core_mode="vector"):
    for start_n in range(0, N_Loop, 1):
        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        # ... online softmax on qk_ub ...
        # ... copy P (NZ format) to L1 ...
        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_MTE3, al.PIPE.PIPE_MTE1)
        al.sync_block_wait("cube", "vector", 2, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        acc = _flash_update(pv_ub, alpha, acc, HEAD_DIM, BLOCK_M)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr_sub, acc.to(Out.type.element_ty))
```

## 5. P 矩阵 ND → NZ 格式转换

Vector softmax 得到 P (ND 格式)，需转为 NZ 分形格式写入 L1 供 Cube 做 PV matmul：

```
ND (BLOCK_M//2, BLOCK_N)
  → reshape → (BLOCK_M//2, BLOCK_N//16, 16)
  → permute [1,0,2] → (BLOCK_N//16, BLOCK_M//2, 16)
  → reshape → (BLOCK_N//16, BLOCK_M//32, 16, 16)   ← NZ 分形格式
```

两个 sub-vector 核通过 `bl.subview` 各自写入 L1 的不同区域，合起来构成完整的 P 矩阵：

```python
p_l1_sub = bl.subview(
    p_l1,
    [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
    [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
    [1, 1, 1, 1],
)
p_nz = p_nz_tmp.reshape(BLOCK_N // 16, BLOCK_M // 32, 16, 16)
al.copy(bl.to_buffer(p_nz, al.ascend_address_space.UB), p_l1_sub)
```

## 6. 编译选项

```python
_attn_fwd[grid](
    ...,
    debug=True,
    disable_auto_inject_block_sync=True,
    vf_merge_level=1,
)
```

`disable_auto_inject_block_sync=True` 是**必须的**：Flash Attention 的同步顺序由手动 `sync_block_set/wait` 精确控制，自动注入会导致死锁或数据竞争。

## 7. 注意事项

1. **fixpipe 只能在 cube scope 内调用**，src 必须是 L0C tensor（`tl.dot` 的结果）
2. **`bl.to_tensor` 在 vector scope 内使用**读取 fixpipe 写入的 UB 数据；在 cube scope 内可用于读取 L1 数据
3. **event_id 范围 0~15**，不同共享资源必须使用不同 event_id
4. **ROW_SPLIT 模式下，UB buffer 的 shape 应为 `(BLOCK_M // 2, ...)`**，因为每个 sub-vector 只看到一半
5. **al.copy / al.fixpipe 仅 A5 可用**
6. **set/wait 必须严格配对且计数平衡**
