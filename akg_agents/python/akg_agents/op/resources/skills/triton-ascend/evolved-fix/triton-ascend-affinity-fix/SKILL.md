---
name: triton-ascend-affinity-fix
description: triton-ascend Cube/Vector 亲和写法常见问题修复：纯 matmul 误用亲和、单一 buffer 跨迭代被覆盖（WAW）、bl.alloc shape 必须 constexpr、fixpipe dst 必须是 buffer
category: fix
version: "1.0.0"
metadata:
  case_type: fix
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  requires_affinity: true
---

# Cube/Vector 亲和写法问题修复

---

## 1. 单一 buffer 在外层 loop 中被反复覆盖（WAW，最常见的精度对不齐根因）

### 现象
Cube scope 在外层 loop 里反复 fixpipe 写同一个 `c_ub`，Vector scope 也在
外层 loop 里反复读同一个 `c_ub`。结果：所有迭代输出值都等于**最后一次写**
的内容。验证报 `err_cnt=XXXX`，最大误差极大。

### 错误代码

```python
@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, ...):
    pid = tl.program_id(0)
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)  # 仅 1 份

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, MAX_CORES):
            acc = tl.dot(a, b, ...)
            al.fixpipe(acc, c_ub, ...)               # 第 i 次写

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, MAX_CORES):
            mm_result = bl.to_tensor(c_ub)            # 永远读到最后一次写
            tl.store(out_block_ptr, mm_result.to(tl.float32))
```

### 根因
- 这两个 `with al.scope` **不是协程并发**，是顺序生成 IR：先生成 cube
  域整段循环（N 次写到同一份 `c_ub`），再生成 vector 域整段循环
  （N 次都从同一份 `c_ub` 读）。
- 等 vector loop 开始读时，`c_ub` 里只剩 cube loop 最后一次写入的 tile。
- 这是 **WAW（写覆盖）**。同步原语 `al.sync_block_set/wait` 解决的是 RAW
  （"读发生在写之后"），它**不能恢复被覆盖掉的历史值**——历史值在物理
  上已经被覆盖。

### 修复：cube 每出一个 tile 后立刻一对 set/wait，复用 c_ub 但保证 vector 已读完

```python
with al.scope(core_mode="cube"):
    for block_idx in ...:
        al.fixpipe(acc, c_ub, ...)
        al.sync_block_set ("cube",   "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        al.sync_block_wait("vector", "cube",   1, al.PIPE.PIPE_V,   al.PIPE.PIPE_FIX)

with al.scope(core_mode="vector"):
    for block_idx in ...:
        al.sync_block_wait("cube",   "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
        tl.store(...)
        al.sync_block_set ("vector", "cube",   1, al.PIPE.PIPE_V,   al.PIPE.PIPE_FIX)
```
---

## 2. `bl.alloc` 的 shape 必须是编译期常量（constexpr）

### 现象

```
TypeError: get_buffer_ty(): incompatible function arguments. ...
Invoked with: <ir.builder>, [64, <triton.language.core.tensor>],
              <ir.type>, <ir.attribute>
```

注意 shape 列表里出现了 `<triton.language.core.tensor>`——某个维度被解析为运行时
tensor，而不是常量。

### 错误代码

```python
@triton.jit
def kernel(..., n_routed_experts, ...):           # n_routed_experts 是 runtime 整型
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, n_routed_experts), al.ascend_address_space.UB)
```

### 根因
`bl.alloc(etype, shape, address_space)` 下沉到 MLIR 的 `builder.get_buffer_ty(shape, ...)`，要求 shape 元素都是 `int`。`@triton.jit` 把不带 `tl.constexpr` 注解的位置参数当成 runtime 值，传进来变成 `tl.tensor`。

### 修复 A：让 shape 用到的符号都标 `tl.constexpr`

```python
@triton.jit
def kernel(..., n_routed_experts: tl.constexpr, ...):   # 编译期常量
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, n_routed_experts), al.ascend_address_space.UB)
```

代价：每个不同 `n_routed_experts` 值触发一次 JIT 重编译。

### 修复 B：引入新的 BLOCK 常量，与 runtime 维度解耦

```python
@triton.jit
def kernel(..., n_routed_experts, BLOCK_E: tl.constexpr, ...):   # n_routed_experts 仍 runtime
    grad_router_logits_ub = bl.alloc(
        tl.float32, (BLOCK_M, BLOCK_E), al.ascend_address_space.UB)
    for e_start in range(0, n_routed_experts, BLOCK_E):
        e_offsets = e_start + tl.arange(0, BLOCK_E)
        e_mask = e_offsets < n_routed_experts
        ...
```

同样的 constexpr 约束适用于 `bl.subview` 的 offsets/sizes/strides、`bl.to_tensor(target_shape=...)`、`bl.to_buffer(...)` 的隐式 shape。

---

## 3. 纯 matmul 错用亲和写法 → 精度大误差

### 现象

把"无后处理 GEMM"（例如 `grad_fc2_weight = grad_output.T @ gelu_output`、
`grad_hidden_state = grad_fc1_output @ fc1_weight`）写成了亲和形式：cube
端 fixpipe 到 UB，vector 端 `bl.to_tensor + tl.store` 写 GM。验证时
**单算 grad_fc2_bias 能 pass，一加上 grad_fc2_weight 就出现量级误差**。

### 错因

无后处理 GEMM 不需要 vector 介入。绕一圈 cube→UB→vector→GM 的代价：

1. 多一次 L0C→UB→GM 中转搬运；
2. 多一对 cube/vector sync 事件；

### 修复：改回原生 Triton

```python
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """无后处理 GEMM：不要写 al.scope / al.fixpipe / bl.alloc。"""
    pid = tl.program_id(0)
    for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
        block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
        # A_blk/B_blk/C_blk: tl.make_block_ptr 标准三件套 (见 guide §6.1)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for _ in range(K_LOOP):
            a = tl.load(A_blk, boundary_check=(0, 1), padding_option="zero")
            b = tl.load(B_blk, boundary_check=(0, 1), padding_option="zero")
            acc = tl.dot(a, b, acc)
            A_blk = tl.advance(A_blk, (0, BLOCK_K))
            B_blk = tl.advance(B_blk, (BLOCK_K, 0))
        tl.store(C_blk, acc, boundary_check=(0, 1))
```