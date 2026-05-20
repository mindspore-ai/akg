---
name: triton-ascend-a5-api
description: "Atlas A5 (Ascend950) 专属 Cube/Vector 协同编程接口。涵盖 Buffer Language (bl.alloc/to_buffer/to_tensor/subview) 和 Ascend Language (al.scope/fixpipe/sync_block_set/sync_block_wait/sub_vec_id/copy) 的完整用法与同步操作。适用于需要在 A5 硬件上进行 Cube 计算后交给 Vector 做后处理（如 bias/relu/softmax）的高性能内核编写场景。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  operator_type: "all"
---

# A5 Cube/Vector 协同编程接口

> 本文档中的 `al.fixpipe`、`al.copy` 等接口仅在 A5 硬件 (Ascend950) 上可用。

## 1. 导包要求

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al   # Ascend Language
import triton.extension.buffer.language as bl        # Buffer Language
```

## 2. Buffer Language (bl) 接口

### 2.1 `bl.alloc(dtype, shape, address_space)` — 分配片上 buffer

```python
# 在 UB 上分配一块 float32 的 [64, 128] buffer
c_ub = bl.alloc(tl.float32, (64, 128), al.ascend_address_space.UB)

# 在 L1 上分配 NZ 格式的 buffer（用于存储 P 矩阵的中间结果）
p_l1 = bl.alloc(tl.float16, (BLOCK_N // 16, BLOCK_M // 16, 16, 16), al.ascend_address_space.L1)
```

**地址空间选项**: `al.ascend_address_space.UB` / `al.ascend_address_space.L1` / `al.ascend_address_space.L0C` / `al.ascend_address_space.L0A` / `al.ascend_address_space.L0B`

**硬规则**: `shape` 的每一维都必须是编译期常量。不要把 runtime
tensor（例如未标注 `tl.constexpr` 的 `n_routed_experts`）直接放进 shape。
需要动态维度时，引入 `BLOCK_*: tl.constexpr` 作为 buffer shape，再用 mask
处理真实边界。

### 2.2 `bl.to_tensor(buffer)` — buffer 转 tensor（供 Vector 计算）

```python
# 在 vector scope 中读取 UB buffer 内容
with al.scope(core_mode="vector"):
    c = bl.to_tensor(c_ub)       # 转为 tensor 后可做向量运算
    result = c + bias[None, :]   # 正常的 tl 运算
```

### 2.3 `bl.to_buffer(tensor, address_space)` — tensor 转 buffer（作为 fixpipe/copy 目标）

```python
# fixpipe 的 dst 参数必须是 buffer
al.fixpipe(acc, bl.to_buffer(c_ub, al.ascend_address_space.UB), ...)

# copy 的 src/dst 也必须是 buffer
al.copy(bl.to_buffer(src_tensor, al.ascend_address_space.UB), dst_l1_buffer)
```

### 2.4 `bl.subview(buffer, offsets, sizes, strides)` — buffer 切片

```python
# 对 L1 buffer 取子视图（用于 sub_vec_id 分块）
p_l1_sub = bl.subview(
    p_l1,
    [0, sub_vec_id * ((BLOCK_M // 2) // 16), 0, 0],
    [BLOCK_N // 16, (BLOCK_M // 2) // 16, 16, 16],
    [1, 1, 1, 1]
)
```

## 3. Ascend Language (al) 接口

### 3.1 `al.scope(core_mode=...)` — 指定 Cube/Vector 执行域

```python
with al.scope(core_mode="cube"):
    # 此区域内的代码运行在 Cube Core 上
    acc = tl.dot(a, b)   # GEMM
    al.fixpipe(acc, c_ub, ...)

with al.scope(core_mode="vector"):
    # 此区域内的代码运行在 Vector Core 上
    c = bl.to_tensor(c_ub)
    result = tl.exp(c)
    tl.store(out_ptr, result)
```

**注意**: Cube scope 内只能做 `tl.dot` / `tl.load`（走 L0A/L0B）/ `al.fixpipe` 等。Vector scope 内做 element-wise / reduce / store。

### 3.2 `al.fixpipe(src, dst, dma_mode, dual_dst_mode)` — L0C → UB 搬运（A5 专属）

```python
al.fixpipe(
    acc,                                    # src: L0C 上的 tensor（tl.dot 结果）
    bl.to_buffer(c_ub, al.ascend_address_space.UB),  # dst: UB buffer
    al.FixpipeDMAMode.NZ2ND,              # DMA 模式
    al.FixpipeDualDstMode.ROW_SPLIT,      # 双目标模式
)
```

**DMA 模式**:
| 模式 | 说明 |
|------|------|
| `NZ2ND` | NZ 格式转 Normal Dense（最常用） |
| `NZ2DN` | NZ 格式转 DN |
| `NZ2NZ` | 保持 NZ 格式 |

**双目标模式（dual_dst_mode）**:
| 模式 | 说明 |
|------|------|
| `NO_DUAL` | 不拆分，整块写入 |
| `ROW_SPLIT` | 按行拆成两半，分别给 2 个 sub-vector core |
| `COLUMN_SPLIT` | 按列拆成两半 |

**对齐约束（float32）**:
- 最后一维必须 8 对齐
- `ROW_SPLIT` / `COLUMN_SPLIT` 时，最后一维必须 32 对齐
- `NZ2DN` 时，第一维必须 8 对齐

**对齐约束（float16/bfloat16）**:
- 最后一维必须 16 对齐

### 3.3 `al.sync_block_set / al.sync_block_wait` — 跨核同步原语

```python
al.sync_block_set(sender, receiver, event_id, sender_pipe, receiver_pipe)
al.sync_block_wait(sender, receiver, event_id, sender_pipe, receiver_pipe)
```

**参数说明**:
| 参数 | 说明 |
|------|------|
| `sender` / `receiver` | `"cube"` 或 `"vector"`，定义事件通道方向 |
| `event_id` | 0~15，同方向的不同资源必须使用不同 event_id |
| `sender_pipe` | SET 时：该管线排空后才发出事件（确保数据就绪） |
| `receiver_pipe` | WAIT 时：该管线被阻塞直到事件到达（确保依赖满足） |

**基本用法**:
```python
# Cube 完成 fixpipe 后通知 Vector（PIPE_FIX 排空后 set，PIPE_V 被解除阻塞）
al.sync_block_set("cube", "vector", event_id, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# Vector 等待 Cube 的 fixpipe 完成
al.sync_block_wait("cube", "vector", event_id, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)

# Vector 计算完成后通知 Cube（PIPE_V 排空后 set，PIPE_FIX 被解除阻塞）
al.sync_block_set("vector", "cube", event_id, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# Cube 等待 Vector 释放 UB
al.sync_block_wait("vector", "cube", event_id, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

### 3.3.1 PIPE 枚举值详解

Ascend NPU 内部有多条独立硬件管线并行执行不同操作，PIPE 参数精确控制同步点：

| PIPE 枚举 | 对应硬件单元 | 职责 | 典型使用场景 |
|---|---|---|---|
| `PIPE_V` | Vector Core 计算管线 | element-wise / reduce / 向量运算 | softmax、epilogue、flash_update |
| `PIPE_FIX` | Fixpipe DMA 管线 | L0C → UB 的 NZ↔ND 格式搬运 | `al.fixpipe` 后 set，fixpipe 前 wait |
| `PIPE_MTE1` | Memory Transfer Engine 1 | **外存 → 片上**搬运（HBM→L1/L0） | `tl.load`，cube 释放 L1 后 set |
| `PIPE_MTE2` | Memory Transfer Engine 2 | 通用 DMA | 较少直接使用 |
| `PIPE_MTE3` | Memory Transfer Engine 3 | **片上 → 片上/外存**搬运 | `al.copy`(UB→L1)、`tl.store`(UB→HBM) |
| `PIPE_M` | Cube Core (Matrix) 管线 | 矩阵乘法 `tl.dot` | 较少直接使用（通过 PIPE_FIX 间接同步） |
| `PIPE_S` | Scalar Core 管线 | 标量计算 | 较少使用 |
| `PIPE_ALL` | 全部管线 | 全部排空 | debug barrier |

### 3.3.2 sender_pipe / receiver_pipe 组合规则

不同的 sender→receiver 方向对应不同的 PIPE 组合：

| 方向 | 典型 sender_pipe | 典型 receiver_pipe | 语义 |
|---|---|---|---|
| cube→vector (fixpipe 数据就绪) | `PIPE_FIX` | `PIPE_V` | fixpipe 完成后 vector 可读 UB |
| vector→cube (UB 已释放) | `PIPE_V` | `PIPE_FIX` | vector 计算完后 cube 可 fixpipe 覆写 UB |
| cube→vector (L1 已释放) | `PIPE_MTE1` | `PIPE_MTE3` | cube 读完 L1 后 vector 可 copy 覆写 L1 |
| vector→cube (L1 数据就绪) | `PIPE_MTE3` | `PIPE_MTE1` | vector copy 到 L1 后 cube 可读 L1 |

### 3.3.3 关键规则

1. **同一 (sender, receiver, event_id) 通道的 set/wait 必须计数平衡**，否则死锁或数据竞争
2. **不同共享资源必须使用不同 event_id**，避免事件冲突

### 3.4 `al.sub_vec_id()` — 获取子 vector 核 ID

```python
with al.scope(core_mode="vector"):
    sub_vec_id = al.sub_vec_id()  # 返回 0 或 1

    # 用 sub_vec_id 计算当前 vector 核负责的行偏移
    OUT_block_ptr = tl.make_block_ptr(
        base=OUT + stride_m * sub_vec_id * (BLOCK_M // 2),
        ...
    )
```

当 `fixpipe` 使用 `ROW_SPLIT` 时，BLOCK_M 行被拆成两半：
- sub_vec_id=0 的核处理上半部分 [0, BLOCK_M//2)
- sub_vec_id=1 的核处理下半部分 [BLOCK_M//2, BLOCK_M)

### 3.5 `al.copy(src_buffer, dst_buffer)` — UB→UB/L1 搬运（A5 专属）

```python
# UB -> L1（用于把 softmax 结果送到 L1 供 Cube 做 PV matmul）
al.copy(bl.to_buffer(p_nz, al.ascend_address_space.UB), p_l1_sub)
```

**约束**: src 必须在 UB，dst 必须在 UB 或 L1，shape/dtype 必须相同。

> `al.copy_from_ub_to_l1` 已废弃，请统一使用 `al.copy`。

## 4. 注意事项

1. **fixpipe 只能在 cube scope 内调用**，src 必须是 L0C tensor（`tl.dot` 的结果）
2. **`bl.to_tensor` 在 vector scope 内使用** 读取 fixpipe 写入的 UB 数据；在 cube scope 内可用于读取 L1 数据
3. **event_id 范围 0~15**，不同共享资源必须使用不同 event_id
4. **ROW_SPLIT 模式下，UB buffer 的 shape 应为 `(BLOCK_M // 2, ...)`**，因为每个 sub-vector 只看到一半
5. **al.copy / al.fixpipe 仅 A5 可用**
6. **set/wait 必须严格配对且计数平衡**，Ping-Pong 下需要 prefree + postwait 补齐首尾
7. **`al.compile_hint(tensor, name)` 标记跨迭代存活的变量**（如 alpha/alpha_pong）
8. **`al.copy_from_ub_to_l1` 已废弃**，请统一使用 `al.copy`
9. **`bl.alloc` / `bl.subview` / `bl.to_tensor(target_shape=...)` 的 shape 参数必须是 constexpr**
