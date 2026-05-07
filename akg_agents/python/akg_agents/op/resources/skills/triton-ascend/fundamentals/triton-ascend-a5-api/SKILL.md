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

## 4. 同步用例

### 4.1 串行场景（单 buffer，Cube/Vector 严格交替）

适用于简单的 GEMM + 后处理（如 matmul + bias + relu），或多阶段串行流水。
每个共享 buffer 只有一份，同一时刻只有一侧在使用。

```python
@triton.jit
def kernel(...):
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            acc = tl.dot(a, b)
            al.fixpipe(acc, bl.to_buffer(c_ub, al.ascend_address_space.UB),
                       al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            c = bl.to_tensor(c_ub)
            result = c + bias[None, :]
            tl.store(out_ptr, result)
            al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

**事件平衡（N 个 tile）**:
- event 0: cube set N 次, vector wait N 次
- event 2: vector set N 次, cube wait N 次

### 4.2 Ping-Pong 流水（双 buffer）

#### 4.2.1 核心思想

串行模式下 Cube 每完成一步都要停下来等 Vector 处理完才能继续。Ping-Pong 双缓冲让 Cube 和 Vector 交替使用两套 buffer，使两者可以重叠执行：

```
串行:  Cube写A → 等Vector读A完 → Cube写A → 等Vector读A完 → ...
流水:  Cube写A → Cube写B → Cube写A → Cube写B → ...
              Vector读A → Vector读B → Vector读A → ...
```

#### 4.2.2 信号量语义

`sync_block_set` / `sync_block_wait` 基于**信号量**机制：
- `set` = 信号量 **+1**（资源已释放 / 数据已就绪）
- `wait` = 信号量 **-1 且阻塞**（信号量=0 时阻塞，>0 时消费一个信用并继续）

同一 `(sender, receiver, event_id)` 通道上，kernel 全生命周期内的 **总 SET 次数必须等于总 WAIT 次数**。

#### 4.2.3 prefree 与 postwait

Ping-Pong 流水的核心难点在于首尾：循环第一次迭代时消费者尚未执行释放操作，但生产者已经需要 wait 获得写入许可。

**prefree（预释放）**：消费者在循环开始前执行 2 次 set（对应 ping/pong 两个 buffer），向信号量注入初始信用，使生产者第一次进入循环可以直接写入。

**postwait（后等待）**：循环最后两次迭代中消费者产生的 set 信号没有被循环体内的 wait 消耗，生产者在循环结束后执行 2 次 wait 消耗剩余信号，保证信号量归零。

```python
# ——— 消费者预释放 2 个 buffer 的初始信用 ———
@triton.jit
def vec_prefree_ub():
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)  # ping
    al.sync_block_set("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)  # pong

# ——— 生产者后等待消耗尾部多余信号 ———
@triton.jit
def cube_postwait_ub():
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
    al.sync_block_wait("vector", "cube", 2, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

#### 4.2.4 每个共享资源一对事件

每个 ping-pong 共享资源需要 **2 个事件通道**（1 个 "数据就绪"、1 个 "已释放"）：

| 资源 | "数据就绪" 事件 | "已释放" 事件 |
|------|---------------|-------------|
| qk_ub (UB) | event 0: cube→vec (PIPE_FIX→PIPE_V) | event 2: vec→cube (PIPE_V→PIPE_FIX) |
| p_l1 (L1) | event 4: vec→cube (PIPE_MTE3→PIPE_MTE1) | event 6: cube→vec (PIPE_MTE1→PIPE_MTE3) |
| pv_ub (UB) | event 8: cube→vec (PIPE_FIX→PIPE_V) | event 10: vec→cube (PIPE_V→PIPE_FIX) |

"数据就绪" = 生产者写完后 set，消费者使用前 wait
"已释放" = 消费者用完后 set，生产者覆写前 wait

#### 4.2.5 批次循环编排 (PIPE_STAGES)

`PIPE_STAGES=2` 表示每批处理 2 个 block，配合双缓冲交替使用 ping/pong：

```python
with al.scope(core_mode="cube"):
    cube_prefree_p_l1()                                  # 预释放 p_l1
    for batch_start in range(0, n_loop, PIPE_STAGES):
        batch_size = min(PIPE_STAGES, n_loop - batch_start)
        for i in range(batch_size):
            _qk_matmul(q, K_ptr, qk_ub_ping, qk_ub_pong, ..., sid)
            sid += 1
        for i in range(batch_size):
            _pv_matmul(p_l1_ping, p_l1_pong, pv_ub_ping, pv_ub_pong, V_ptr, ..., pvid)
            pvid += 1
    cube_postwait_s_ub()                                 # 后等待 qk_ub
    cube_postwait_pv_ub()                                # 后等待 pv_ub

with al.scope(core_mode="vector"):
    vec_prefree_s_ub()                                   # 预释放 qk_ub
    vec_prefree_pv_ub()                                  # 预释放 pv_ub
    for batch_start in range(0, n_loop, PIPE_STAGES):
        batch_size = min(PIPE_STAGES, n_loop - batch_start)
        for i in range(batch_size):
            m_i, l_i, alpha, alpha_pong = _softmax(..., sid)
            sid += 1
        for i in range(batch_size):
            acc = _flash_update(pv_ub_ping, pv_ub_pong, alpha, alpha_pong, acc, ..., pvid)
            pvid += 1
    vec_postwait_p_l1()                                  # 后等待 p_l1
```

**时间线示意（n_loop=4）**:
```
Cube:   |QK0 QK1|PV0 PV1|QK2 QK3|PV2 PV3|postwait
Vector: |prefree |SM0 SM1|UP0 UP1|SM2 SM3|UP2 UP3|postwait
```

其中 QK0 写入 `qk_ub_ping`，QK1 写入 `qk_ub_pong`；SM0 从 `qk_ub_ping` 读，SM1 从 `qk_ub_pong` 读。两侧通过 event 0/2 协调，实现写入与读取的重叠。

#### 4.2.6 信号量平衡验证表

对每个 event_id 统计全生命周期的 SET/WAIT 次数（n = n_loop）：

| Event | 方向 | 总 SET | 总 WAIT | 平衡 |
|-------|------|--------|---------|------|
| 0 | cube→vec | n (qk_matmul) | n (softmax) | ✓ |
| 2 | vec→cube | 2(prefree) + n(softmax) | n(qk_matmul) + 2(postwait) | ✓ |
| 4 | vec→cube | n (softmax) | n (pv_matmul) | ✓ |
| 6 | cube→vec | 2(prefree) + n(pv_matmul) | n(softmax) + 2(postwait) | ✓ |
| 8 | cube→vec | n (pv_matmul) | n (flash_update) | ✓ |
| 10 | vec→cube | 2(prefree) + n(flash_update) | n(pv_matmul) + 2(postwait) | ✓ |


## 5. 注意事项

1. **fixpipe 只能在 cube scope 内调用**，src 必须是 L0C tensor（`tl.dot` 的结果）
2. **`bl.to_tensor` 在 vector scope 内使用** 读取 fixpipe 写入的 UB 数据；在 cube scope 内可用于读取 L1 数据
3. **event_id 范围 0~15**，不同共享资源必须使用不同 event_id
4. **ROW_SPLIT 模式下，UB buffer 的 shape 应为 `(BLOCK_M // 2, ...)`**，因为每个 sub-vector 只看到一半
5. **al.copy / al.fixpipe 仅 A5 可用**
6. **set/wait 必须严格配对且计数平衡**，Ping-Pong 下需要 prefree + postwait 补齐首尾
7. **`al.compile_hint(tensor, name)` 标记跨迭代存活的变量**（如 alpha/alpha_pong）
8. **`al.copy_from_ub_to_l1` 已废弃**，请统一使用 `al.copy`
