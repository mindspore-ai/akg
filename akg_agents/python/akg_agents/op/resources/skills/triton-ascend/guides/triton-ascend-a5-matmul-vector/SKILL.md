---
name: triton-ascend-a5-matmul-vector
description: "适用于 A5（Ascend950）Cube/Vector 协同编程的 MatMul + Vector 后处理融合优化指南。当算子的核心计算是矩阵乘法后接逐元素操作（如 bias 加法、ReLU 激活、残差加、量化等）时应选择此指南。本指南采用两段式调度：一个 cube scope 整段循环 + 一个 vector scope 整段循环 + 单 buffer + 一对显式同步事件。覆盖 Cube/Vector 数据流、ROW_SPLIT 拆分、sub_vec_id 索引、显式 sync_block_set/wait 配对、plain matmul kernel 推荐写法、关键约束速查等。不适用于纯 Vector 逐元素运算、也不适用于无后处理的纯 MatMul。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  note: "A5(Ascend950) Cube/Vector 亲和接口 MatMul + Vector 后处理融合"
  operator_type: "matmul"
---

# MatMul + Vector 协同编程优化指南

## 0. 何时用亲和写法

A5 亲和 API（`al.scope` / `al.fixpipe` / `al.sync_block_set/wait` / `bl.alloc`）的真实收益**不来自 fixpipe 本身**，而来自它能让 GEMM 的结果"不落 GM、直接交给 vector 做后处理"——省掉一份 (M, N) 量级的 GM 中间张量、并打开 cube/vector 流水重叠。

这就决定了亲和写法对一种 kernel 形态会有收益：**CV 融合**——一个 kernel 里既有 `tl.dot`（cube），又有可融合的 vector 后处理（GELU / ReLU / Sigmoid / Softmax / Bias-add / Scale / Mask / Reduce / 量化等）。

### 0.1 三条硬性规则

#### 规则 A：纯 MatMul **绝对不要**用亲和接口实现 — 必须用原生 Triton

> 适用：算子的 kernel 体内**只有** `tl.dot`，**没有任何**可融合的 vector 后处理。例如：
> - `matmul` 算子（`Y = X @ W`）；
> - `grad_fc2_weight = grad_output.T @ gelu_output`、
>   `grad_fc1_weight = grad_fc1_output.T @ hidden_state`、
>   `grad_hidden_state = grad_fc1_output @ fc1_weight`

**必须**写成原生 Triton —— `tl.make_block_ptr` + `tl.load` + `tl.dot` + `tl.store`，**绝对禁止**写入 `al.scope` / `al.fixpipe` / `bl.alloc` 任何一个。原因：

1. cube 算完的 acc 在 L0C 上是 NZ 格式，原生 `tl.store(GM_block_ptr, acc)` 在 cube 路径下编译器会自动 lower 成**隐式 fixpipe(L0C → GM)** 直写 GM —— 这就是 cube 数据出口最优的硬件指令路径。
2. 如果手动套上 `al.fixpipe(acc, c_ub)` 把数据先搬到 UB、再用 `bl.to_tensor + tl.store` 写 GM，会**多一次** L0C→UB→GM 的中转搬运、**多一对** cube/vector sync 事件，并把 vector 单元无意义地锁住。
3. 同时，UB 中转那条路径在 ROW_SPLIT / sub_vec_id / non-aligned shape 上容易引入精度问题。

纯 matmul kernel 的推荐模板见 6.1。

#### 规则 B：纯 Vector（无 `tl.dot`）使用原生 Triton vector 写法，亲和 API 对它们没有意义

softmax / layernorm / reduce / pure elementwise 等算子都属于这一类。

---

## 1. 适用场景

许多算子的核心计算模式是"矩阵乘法 + 逐元素后处理"，例如：

- **Linear + bias**：`Y = X @ W + bias`
- **MatMul + ReLU**：`Y = ReLU(X @ W)`
- **MatMul + 残差加**：`Y = X @ W + residual`
- **MatMul + GELU**：`Y = GELU(X @ W)`
- **MatMul + 量化**：`Y = quantize(X @ W)`

这类算子在 Atlas A5 上可以利用 Cube/Vector 协同编程实现高效融合：Cube 负责矩阵乘法，Vector 负责后处理，通过 `al.fixpipe` 在片上传递中间结果，避免数据回写 GM 再读取的开销。

## 2. 调度结构（fused-cv kernel）

**关键设计原则**：不要把 cube 和 vector 写成"逐 block 交错穿插"。正确写法是 **两段式**：

- cube scope：`for tile in [0..N): dot(K-loop) → fixpipe → c_ub; sync_set(cube→vector, EVT0); sync_wait(vector→cube, EVT1)`；
- vector scope：`for tile in [0..N): sync_wait(cube→vector, EVT0); read c_ub → 后处理 → store GM; sync_set(vector→cube, EVT1)`。

EVT0 = data-ready，EVT1 = buffer-free。单 buffer `c_ub` 跨 tile 复用，如果没有 EVT1，cube 第二次 fixpipe 会在 vector 还没读完前覆盖 ub，触发 WAW，结果**所有 tile 的输出都等于最后一次写入的 tile**。

## 3. 数据流

`GM(input/weight) → tl.load (cube) → L0C(fp32 acc) → al.fixpipe(NZ2ND,ROW_SPLIT) → UB c_ub(BM/2, BN) → bl.to_tensor (vector) → 后处理 → tl.store → GM`

## 4. fused-cv kernel 的结构设计

### 4.1 Buffer 分配

```python
c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)
```

- 行维度使用 `BLOCK_M // 2`：`ROW_SPLIT` 模式下每个 sub-vector core 只看到一半行。
- dtype 用 `tl.float32`：Cube L0C 累加器是 fp32，fixpipe 直接搬出。
- shape 必须是 `tl.constexpr`。

> 简化变体：如果不需要 sub-vector 拆分，可以省掉 `FixpipeDualDstMode.ROW_SPLIT`，buffer shape 直接用 `(BLOCK_M, BLOCK_N)`，vector 端不带 `sub_vec_id` 偏移，写法更简单也更稳。

### 4.2 Cube Scope — 矩阵乘法 + fixpipe

```python
with al.scope(core_mode="cube"):
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N

        # A_blk: (M,K), offsets=(block_m*BM, 0), block_shape=(BM, BK), order=(1,0)
        # B_blk: (K,N), offsets=(0, block_n*BN), block_shape=(BK, BN), order=(1,0)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _k in range(K_LOOP):
            a = tl.load(A_blk); b = tl.load(B_blk)
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))

        al.fixpipe(acc, c_ub,
                   al.FixpipeDMAMode.NZ2ND, al.FixpipeDualDstMode.ROW_SPLIT)
        al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

要点：

- `tl.dot(a, b, acc)` 三操作数形式做 K 维 reduce；`acc` 必须显式 `tl.zeros` 初始化（L0C 不保证清零）。
- `al.fixpipe(NZ2ND, ROW_SPLIT)` 把 L0C 的 NZ 分形展开成 ND 行优先并自动按行切给两个 sub-vector。
- 每 tile 一对 set/wait：set 用 `PIPE_FIX → PIPE_V`，wait 用 `PIPE_V → PIPE_FIX`，event id 一定要配对一致。

### 4.3 Vector Scope — 后处理 + 存储

```python
with al.scope(core_mode="vector"):
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
        sub_vec_id = al.sub_vec_id()                    # 0 or 1

        al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        tile = bl.to_tensor(c_ub)                       # (BM/2, BN) fp32
        tile = tl.maximum(tile, 0.0)                    # 后处理（ReLU 示例）

        # C_blk: (M,N), block_shape=(BM/2, BN), order=(1,0)
        # offsets=(block_m*BM + sub_vec_id*(BM/2), block_n*BN)
        tl.store(C_blk, tile.to(C_ptr.dtype.element_ty), boundary_check=(0, 1))

        al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

要点：

- `al.sub_vec_id()` 返回 0 或 1，**输出行偏移必须加 `sub_vec_id * (BLOCK_M // 2)`**，否则两个 sub-vec 写到同一段 GM 上发生覆盖。
- bias 类后处理（broadcast）：`bias_tile = tl.load(bias_ptr + col_off + tl.arange(0, BLOCK_N)); tile = tile + bias_tile[None, :]`。
- 写完 ub 后立刻 `sync_set(EVT_BUF_FREE)`，让 cube 可以进入下一 tile。

### 4.4 fused-cv kernel 的常见设计要点

1. **后处理放在 vector scope**：cube fixpipe 完之后，vector `bl.to_tensor(c_ub)` 拿 tile，做 elementwise / reduction，再 `tl.store` 出去。
2. **后处理涉及到的额外输入**（例如 GELU' 里的 `fc1_output`）由 vector 端用 `tl.load` 直接读 GM，**不必走 cube**。

## 5. 同步事件配对速查

| Event | 方向 | sender_pipe → receiver_pipe | 含义 |
|-------|------|-----------------------------|------|
| 0 | cube → vector | `PIPE_FIX` → `PIPE_V` | "tile 已经在 c_ub 里了" |
| 1 | vector → cube | `PIPE_V` → `PIPE_FIX` | "c_ub 我读完了，可以覆写" |

**配对原则**：set/wait 五元组 `(producer, consumer, event_id, src_pipe, dst_pipe)` 必须完全一致。少 set 或多 wait → 死锁；PIPE 写反 → 命中不到事件。

## 6. plain matmul kernel 的推荐写法（**纯 matmul 不要用亲和**）
### 6.1 推荐模板（无后处理 GEMM）

```python
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """无后处理 GEMM：纯原生 Triton。禁止使用 al.scope / al.fixpipe / bl.alloc。
    传入 stride、BLOCK constexpr、K_LOOP/NUM_BLOCKS/NUM_BLOCKS_N/CORE_NUM。"""
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N

        # A_blk/B_blk/C_blk: tl.make_block_ptr 标准三件套, order=(1,0)
        # A=(M,K)@(block_m*BLOCK_M, 0) (BLOCK_M, BLOCK_K)
        # B=(K,N)@(0, block_n*BLOCK_N) (BLOCK_K, BLOCK_N)
        # C=(M,N)@(block_m*BLOCK_M, block_n*BLOCK_N) (BLOCK_M, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for _k in range(K_LOOP):
            a = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))

        tl.store(C_blk, acc, boundary_check=(0, 1))
```

要点：

- `boundary_check=(0, 1), padding_option="zero"` 必须加，否则 M / N / K 不对齐时 tail block 会读到未定义数据。
- 不要传 `disable_auto_inject_block_sync=True` / `debug=True`——这两个是亲和写法专用，原生 Triton 走标准 lowering。

## 7. 关键约束速查

1. **fixpipe 出入口**：必须在 cube scope 内调用，src 是 `tl.dot` 的 acc（L0C），dst 必须是 `bl.alloc` 的 buffer（UB / L1 / L0x），传 GM block_ptr 报 `TypeError('dst is not of buffer type')`。
2. **`bl.alloc` shape 必须 constexpr**；同理 `bl.subview` / `bl.to_tensor(target_shape=...)` 也是。runtime 整型（如 `n_routed_experts`）会报 `get_buffer_ty()` 错。
3. **set/wait 五元组完全配对**：`(producer, consumer, event_id, src_pipe, dst_pipe)` 任一处不一致就死锁/丢事件；event id 0~15。Cube 与 Vector 两个 for 循环必须遍历完全相同的 block 序列（起点/步长一致）。
4. **`disable_auto_inject_block_sync=True` 仅亲和写法用**；plain matmul / pure vector kernel 一律不传。`al.fixpipe / al.copy / al.scope / bl.alloc` 仅 A5 可用。
