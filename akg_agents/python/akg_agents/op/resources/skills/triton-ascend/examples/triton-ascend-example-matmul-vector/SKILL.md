---
name: triton-ascend-example-matmul-vector
description: "A5（Ascend950）MatMul + Vector 后处理融合的完整 Triton-Ascend 实现示例集合。包含两个完整可运行案例：(1) MatMul + ReLU（fp16）单 kernel CV 融合；(2) Vision MLP + GELU backward —— 一个 fused-cv kernel + 三个 plain matmul kernel + 两个 pure-vector kernel 同文件混合形态。"
category: example
version: "1.0.0"
metadata:
  backend: ascend
  dsl: triton_ascend
  hardware: "Atlas A5"
  operator_type: "matmul"
---

# A5 MatMul + Vector 协同编程 — 完整代码示例

## 1. MatMul + ReLU（fp16）— 单 kernel CV 融合
```python

@triton.jit
def matmul_relu_kernel(A_ptr, B_ptr, C_ptr, ...):
    """参数：M/N/K + NUM_BLOCKS / NUM_BLOCKS_N / CORE_NUM """
    pid = tl.program_id(0)
    K_LOOP: tl.constexpr = (K + BLOCK_K - 1) // BLOCK_K
    c_ub = bl.alloc(tl.float32, (BLOCK_M // 2, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            # A_blk: (M,K)@(block_m*BM, 0) (BM, BK), order=(1,0)
            # B_blk: (K,N)@(0, block_n*BN) (BK, BN), order=(1,0)

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

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            sub_vec_id = al.sub_vec_id()

            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            tile = bl.to_tensor(c_ub)
            tile = tl.maximum(tile, 0.0)                       # ReLU
            # C_blk: (M,N), block_shape=(BM/2, BN), order=(1,0)
            # offsets=(block_m*BM + sub_vec_id*(BM/2), block_n*BN)
            tl.store(C_blk, tile.to(tl.float16), boundary_check=(0, 1))
            al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)
```

要点：本例采用 `(BM/2, BN)` UB buffer + `ROW_SPLIT` + `sub_vec_id`，吃满双 sub-vector；fp16 输出需显式 `.to(tl.float16)`；host 调用必须传 `disable_auto_inject_block_sync=True`。

---

## 2. Vision MLP + GELU backward —— 混合形态（fused-cv + plain matmul + pure vector）

```python
grad_fc2_bias    = grad_output.sum(dim=0)
grad_fc2_weight  = grad_output.t().mm(gelu_output)
grad_gelu_output = grad_output.mm(fc2_weight)
# GELU backward (tanh approximation) ——
grad_fc1_output  = grad_gelu_output * GELU'(fc1_output)
grad_fc1_bias    = grad_fc1_output.sum(dim=0)
grad_fc1_weight  = grad_fc1_output.t().mm(hidden_state)
grad_hidden_state = grad_fc1_output.mm(fc1_weight)
```

7 个操作如何分配到 kernel —— 这是CV融合算子的融合算子选择策略：

| 操作 | kernel 类型 | 对应实现 |
|---|---|---|
| `grad_gelu_output = grad_output @ fc2_weight` 紧接 `grad_fc1_output = grad_gelu_output * GELU'(fc1_output)` | **fused-cv**（合一个 kernel） | K1：cube 算 GEMM，fixpipe 到 UB；vector 端读 UB + 读 `fc1_output` 计算 GELU' + 乘法 + 写 `grad_fc1_output` 出 GM。**省掉一个 (S, I) 的中间张量 `grad_gelu_output` 完整地在 GM 上的写+读。** |
| `grad_fc2_weight = grad_output.T @ gelu_output` | **plain matmul** | K2-1：原生 Triton GEMM，host 用 `grad_output.t().contiguous()` 物化 transpose |
| `grad_fc1_weight = grad_fc1_output.T @ hidden_state` | **plain matmul** | K2-2：原生 Triton GEMM，host 用 `grad_fc1_output.t().contiguous()` 物化 transpose |
| `grad_hidden_state = grad_fc1_output @ fc1_weight` | **plain matmul** | K2-3：原生 Triton GEMM |
| `grad_fc2_bias = sum(grad_output, dim=0)` | **pure vector** | K3：纯 vector reduce |
| `grad_fc1_bias = sum(grad_fc1_output, dim=0)` | **pure vector** | K3：纯 vector reduce |

> 融合选择原则：**有一处值得做亲和** —— `GEMM-2 紧跟 GELU' 乘法` 这条边界省下 (S, I) 的 GM 中转。其他 GEMM 都是无后处理的，用原生 Triton。

```python

@triton.jit
def _gelu_grad(x):
    """GELU'(x) tanh-approx：tanh_out = tl.math.tanh(SQRT_2_OVER_PI*(x+GELU_C*x^3));
    return 0.5*(1+tanh_out) + 0.5*x*(1-tanh_out^2)*SQRT_2_OVER_PI*(1+3*GELU_C*x^2)"""
    ...

# K1: fused (grad_output @ fc2_weight) + (* GELU'(fc1_output)) -> grad_fc1_output
# Cube + Vector 亲和写法，中间结果 grad_gelu_output 不搬进 GM。
@triton.jit
def _k1_fused_gemm_gelu_kernel(GO_ptr, W_ptr, X1_ptr, GFO_ptr, S, H, I_DIM, ...):
    """参数：ptr (grad_output / fc2_weight / fc1_output / grad_fc1_output) + stride
    + BLOCK constexpr + K_LOOP / NUM_BLOCKS / NUM_BLOCKS_N / CORE_NUM。"""
    pid = tl.program_id(0)
    fused_ub = bl.alloc(tl.float32, (BLOCK_M, BLOCK_N), al.ascend_address_space.UB)

    with al.scope(core_mode="cube"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            # A_ptr: (S,H)@(block_m*BM, 0) (BM, BK), B_ptr: (H,I)@(0, block_n*BN) (BK, BN), order=(1,0)

            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for _k in range(K_LOOP):
                a = tl.load(A_ptr, boundary_check=(0, 1), padding_option="zero")
                b = tl.load(B_ptr, boundary_check=(0, 1), padding_option="zero")
                acc = tl.dot(a, b, acc)
                A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
                B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))

            al.fixpipe(acc, fused_ub, al.FixpipeDMAMode.NZ2ND)
            al.sync_block_set("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            al.sync_block_wait("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

    with al.scope(core_mode="vector"):
        for block_idx in range(pid, NUM_BLOCKS, CORE_NUM):
            block_m, block_n = block_idx // NUM_BLOCKS_N, block_idx % NUM_BLOCKS_N
            al.sync_block_wait("cube", "vector", 0, al.PIPE.PIPE_FIX, al.PIPE.PIPE_V)
            grad_gelu_tile = bl.to_tensor(fused_ub)              # (BM, BN) fp32

            # X1_blk / GFO_blk: (S, I_DIM)@(block_m*BM, block_n*BN) (BM, BN), order=(1,0)
            x = tl.load(X1_blk, boundary_check=(0, 1), padding_option="zero")
            grad_fc1_tile = grad_gelu_tile * _gelu_grad(x)
            tl.store(GFO_blk, grad_fc1_tile, boundary_check=(0, 1))

            al.sync_block_set("vector", "cube", 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)

# K2: 三个无后处理 GEMM —— 原生 Triton。禁止 al.scope / al.fixpipe / bl.alloc。
@triton.jit
def _matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, ...):
    """Native Triton GEMM"""
    ...

# K3: 纯 vector。
@triton.jit
def _bias_reduce_kernel(X_ptr, Y_ptr, M, N, stride_xm, stride_xn, ...):
    pid = tl.program_id(0)
    with al.scope(core_mode="vector"):
        for block_n in range(pid, NUM_BLOCKS_N, CORE_NUM):
            n_offsets = block_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for m_start in range(0, M, BLOCK_M):
                m_offsets = m_start + tl.arange(0, BLOCK_M)
                m_mask = m_offsets < M
                offs = m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xn
                acc += tl.load(X_ptr + offs,
                               mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            tl.store(Y_ptr + n_offsets, tl.sum(acc, axis=0), mask=n_mask)

class ModelNew(nn.Module):
    """Serial Cube/Vector affinity version of vision MLP + GELU backward."""

    def forward(self, grad_output, hidden_state, fc1_weight, fc1_bias,
                      fc2_weight, fc2_bias, fc1_output, gelu_output):
        # K1: tail rows 必须 zeros（K2/K3 后续会读这些被跳过的行）
        grad_fc1_output = torch.zeros((S, I), device=device, dtype=torch.float32)
        _k1_fused_gemm_gelu_kernel[(num_cores,)](
            grad_output, fc2_weight, fc1_output, grad_fc1_output, S, H, I, ...,
            debug=True, disable_auto_inject_block_sync=True,
        )

        grad_output_T   = grad_output.t().contiguous()              # (H, S)
        grad_fc1_out_T  = grad_fc1_output.t().contiguous()          # (I, S)
        # K2-1: grad_fc2_weight   (H, I) = grad_output_T  @ gelu_output
        # K2-2: grad_fc1_weight   (I, H) = grad_fc1_out_T @ hidden_state
        # K2-3: grad_hidden_state (S, H) = grad_fc1_output @ fc1_weight
        # 三次调用都用 _matmul_kernel；不传 debug / disable_auto_inject_block_sync。

        # K3:
        # _bias_reduce_kernel(grad_output,      grad_fc2_bias, S, H, ...)  -> (H,)
        # _bias_reduce_kernel(grad_fc1_output,  grad_fc1_bias, S, I, ...)  -> (I,)
        # 两次都传 debug=True, disable_auto_inject_block_sync=True（vector scope 属于亲和写法）

        return grad_hidden_state, grad_fc1_weight, grad_fc1_bias, grad_fc2_weight, grad_fc2_bias
```

### 2.2 性能收益的来源

K1 把 `GEMM-2 + GELU' 乘法` 在片上 (UB) 直接接起来，**消除 (S, I) 中间张量 `grad_gelu_output` 的 GM 写+读**——这就是这个算子上亲和写法的全部收益来源。K2/K3 走各自最优路径，与亲和无关；强行把 K2 改成亲和形式不仅没有收益、还会引入精度问题。
