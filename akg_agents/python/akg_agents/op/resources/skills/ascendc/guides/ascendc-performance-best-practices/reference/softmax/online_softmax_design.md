# Online Softmax / Tiled Softmax 优化设计

## 1. 优化目标

在 Attention 等长序列场景中，Softmax 的输入是完整的 $QK^T$ 矩阵（规模为 $S \times S$）。Naive 实现先计算完整矩阵再逐行做 Softmax，需要 $O(S^2)$ 内存，长序列下 SRAM 放不下。

本优化将 Softmax 从"全量计算后归一化"改为"逐 Tile 计算，维护 running max/sum"的增量算法。核心思想：在 S2 方向分 Tile 计算 $Q_i K_j^T$，跨 Tile 维护当前见过的最大值 $m$ 和累积和 $l$，输出 $O$ 增量更新。**内存复杂度从 $O(S^2)$ 降至 $O(S)$。**

| 指标 | naive | optimized | 收益 |
|------|-------|-----------|------|
| 中间内存占用 | $O(S^2)$，需完整 $QK^T$ 矩阵 | $O(S)$，仅当前 tile + running 状态 | 长序列下内存大幅降低 |
| max/sum 计算 | 双 pass（先 max 再 sum） | **单 pass**，tile 内即时更新 | 天然 Safe Softmax |
| 输出更新 | 最后一次性 prob × V | 每 tile 增量加权累加 | 无需显式分配 prob 矩阵 |

> 适用算子族：`softmax`（含 `softmax`、`log_softmax` 等变体）以及内嵌 Softmax 的 `flash_attention`、`sparse_flash_attention`。

## 2. 架构概览

### 2.1 存储层级与数据流

```
GM (Global Memory)
  │
  │ MTE2 (Q_i, K_j, V_j 从 GM → L1/UB)
  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SRAM / UB                                                                  │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Q_i Tile │ × │  K_j Tile^T │ = │  S_ij Tile   │   │  O_i (output)   │  │
│  │  [B,S1,D]│   │   [D,S2tile]│   │ [B,S1,S2tile]│ → │ [B,S1,D]        │  │
│  └──────────┘   └─────────────┘   └──────────────┘   └─────────────────┘  │
│         │                                              ▲                    │
│         │        ┌─────────────────────────────┐       │                    │
│         └───────→│ Online Softmax (Vector PIPE)│───────┘                    │
│                  │  running m: max so far       │                            │
│                  │  running l: sum of exp       │                            │
│                  │  O_acc: weighted output      │                            │
│                  └─────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
  │
  │ MTE3 (O_i, m_final, l_final → GM / workspace)
  ▼
GM (Output)
```

### 2.2 Tiled 数据流

- **外循环**：沿 S2 维度分 Tile（$j = 0, 1, \dots, N_{tile}-1$），K、V 以统一块大小 $B$ 划分。
- **每 Tile 内**：计算 $S_{ij} = Q_i K_j^T / \sqrt{d}$，然后做 Online Softmax 更新 running 状态。
- **跨 Tile 状态传递**：$m$（running max）和 $l$（running sum）作为向量在 Tile 间传递。Ascend 场景下通常需 GM workspace 保存跨 S2 tile 的中间态。
- **输出 O 增量更新**：每轮用当前 tile 的 Softmax 权重 $P_{ij}$ 对 $V_j$ 做加权，按修正因子 rescale 后累加到 $O_i$。

### 2.3 核心数学：Online Softmax 单 Pass 公式

对当前 Q block $Q_i$ 与 K block $K_j$ 计算：

**Step 1 — Score**
$$S_{ij} = Q_i \times K_j^T / \sqrt{d}$$

**Step 2 — 局部 max 与更新 running max**
$$m_{new} = \max(m_{old}, \ \text{rowmax}(S_{ij}))$$

**Step 3 — 概率指数（中间变量）**
$$P_{ij} = \exp(S_{ij} - m_{new})$$

**Step 4 — 更新 running sum**
$$l_{new} = l_{old} \times \exp(m_{old} - m_{new}) + \text{rowsum}(P_{ij})$$

**Step 5 — 输出 O 增量更新**
$$O_i = O_i \times \frac{l_{old} \times \exp(m_{old} - m_{new})}{l_{new}} + \frac{P_{ij} \times V_j}{l_{new}}$$

等价概括形式：
```
m_new = max(m_old, tile_max)
sum_new = sum_old * exp(m_old - m_new) + tile_sum_exp(tile - m_new)
```
最终 Softmax：$\text{softmax}(x) = \exp(x - m_{final}) / sum_{final}$

### 2.4 事件同步模型

| 事件类型 | 含义 | 用途 |
|---------|------|------|
| `MTE2_V` | MTE2 搬运完成 → 允许 Vector 读取 | Q/K/V tile 数据就绪 |
| `V_MTE3` | Vector 完成 → 允许 MTE3 写回 | O 增量计算完成，可写 GM |
| `V_V` | Vector 完成 → 允许 Vector 继续 | Online softmax 内部依赖 |

## 3. 关键参数配置

```cpp
// Host 侧 TilingData
struct OnlineSoftmaxTiling {
    uint32_t B;            // tile 块大小（统一划分 K、V），通常 64
    uint32_t D;            // head dimension
    uint32_t seqLen;       // 序列长度 S
    uint32_t s2TileNum;    // S2 方向 tile 数量 = ceil(S / B)
};

// Kernel 侧 running 状态（每行一个）
TBuf<QuePosition::VECCALC> runningMaxBuff;   // m: running max
TBuf<QuePosition::VECCALC> runningSumBuff;   // l: running sum
TBuf<QuePosition::VECCALC> outAccBuff;       // O_acc: 加权输出累加器
```

### 3.1 Tile 大小选取原则

| 参数 | 典型值 | 说明 |
|------|--------|------|
| $B$ | 64 / 128 | 统一 tile 块大小，划分 K、V 沿 S2 维度。需 fit UB |
| D | 64 / 128 | head dimension，由模型决定 |

**Ascend Cube 粒度对齐约束**：

Tile 大小必须对齐 Cube 单元 matmul 粒度，通常为 **16×16 或 32×32**。若 $B$ 未对齐，CUBE 核心需额外 padding。

```
B = align_up(preferred_B, cube_granularity)   // typically 16 or 32
```

### 3.2 内存预算

Online Softmax 将内存复杂度从 $O(S^2)$ 降至 $O(S)$，但 score tile（$B \times B$）、Q tile、K/V tile 同时存在于 UB 时，需确保总占用小于 UB 容量。典型配置下约 65KB，安全。

## 4. 核心计算循环

### 4.1 naive 版本（优化前）

```cpp
// 阶段 1：完整计算 QK^T（S×S 矩阵），需要 O(S^2) 空间
for (uint32_t i = 0; i < S; i++) {
    for (uint32_t j = 0; j < S; j++) {
        scoreGm[i * S + j] = ComputeScore(QGm, KGm, i, j);
    }
}

// 阶段 2：逐行 Softmax，需两次遍历（先求 max，再求 exp/sum）
for (uint32_t i = 0; i < S; i++) {
    float maxVal = -INFINITY;
    for (uint32_t j = 0; j < S; j++) {
        maxVal = max(maxVal, scoreGm[i * S + j]);
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < S; j++) {
        sum += exp(scoreGm[i * S + j] - maxVal);
    }
    for (uint32_t j = 0; j < S; j++) {
        probGm[i * S + j] = exp(scoreGm[i * S + j] - maxVal) / sum;
    }
}

// 阶段 3：prob × V
for (uint32_t i = 0; i < S; i++) {
    for (uint32_t d = 0; d < D; d++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < S; j++) {
            acc += probGm[i * S + j] * VGm[j * D + d];
        }
        OGm[i * D + d] = acc;
    }
}
```

### 4.2 optimized 版本（优化后）：Tiled Online Softmax

```cpp
// Init：分配 running 状态 buffer（每行一个）
pipe->InitBuffer(runningMaxBuff, B * sizeof(COMPUTE_T));   // m
pipe->InitBuffer(runningSumBuff, B * sizeof(COMPUTE_T));   // l
pipe->InitBuffer(outAccBuff, B * D * sizeof(COMPUTE_T));   // O_acc

LocalTensor<COMPUTE_T> mUb = runningMaxBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> lUb = runningSumBuff.Get<COMPUTE_T>();
LocalTensor<COMPUTE_T> oUb = outAccBuff.Get<COMPUTE_T>();

// 初始化 running 状态
Duplicate(mUb, FLOAT_NEG_INF, B);   // m = -inf
Duplicate(lUb, 0.0f, B);            // l = 0
Duplicate(oUb, 0.0f, B * D);        // O = 0

// MTE2 加载 Q_i tile（固定在 UB）
LocalTensor<T> qUb = LoadQTile(QGm, i * B, B, D);

// S2 方向分 tile 循环（Online Softmax 核心）
for (uint32_t j = 0; j < s2TileNum; j++) {
    // 1. MTE2 加载 K_j, V_j tiles
    LocalTensor<T> kUb = LoadKTile(KGm, j * B, B, D);
    LocalTensor<T> vUb = LoadVTile(VGm, j * B, B, D);

    // 2. 计算 S_ij = Q_i * K_j^T  (CUBE / Vector)
    LocalTensor<COMPUTE_T> sUb = ComputeScore(qUb, kUb, B, D);

    // 3. Online Softmax 更新（公式见 2.3）
    OnlineSoftmaxUpdate(sUb, vUb, mUb, lUb, oUb, B, D);

    // 4. 可选：MTE3 将中间结果写回 GM workspace（跨 tile 状态传递）
}

// 所有 tile 结束后：O_final = O_acc / l_final
ElemwiseDiv(oUb, lUb, B, D);

// MTE3 写回最终输出
WriteBackOGm(OGm, oUb, i * B, B, D);
```

### 4.3 Online Softmax Update 伪代码（Vector 核心）

```cpp
void OnlineSoftmaxUpdate(LocalTensor<COMPUTE_T> sUb,   // [B, B] score tile S_ij
                         LocalTensor<T> vUb,           // [B, D] V tile V_j
                         LocalTensor<COMPUTE_T> mUb,   // [B] running max
                         LocalTensor<COMPUTE_T> lUb,   // [B] running sum
                         LocalTensor<COMPUTE_T> oUb,   // [B, D] output acc O_i
                         uint32_t B, uint32_t D) {
    for (uint32_t r = 0; r < B; r++) {
        // Step 1: m_new = max(m_old, rowmax(S_ij))
        float m_old = mUb[r];
        float m_local = ReduceMax(sUb[r * B], B);
        float m_new = max(m_old, m_local);

        // Step 2: P_ij = exp(S_ij - m_new)
        // Step 3: l_new = l_old * exp(m_old - m_new) + rowsum(P_ij)
        float l_old = lUb[r];
        float scale = exp(m_old - m_new);
        float sum_exp = 0.0f;

        LocalTensor<COMPUTE_T> pUb = tempBuff.Get<COMPUTE_T>();
        for (uint32_t c = 0; c < B; c++) {
            float p_val = exp(sUb[r * B + c] - m_new);
            pUb[c] = p_val;
            sum_exp += p_val;
        }
        lUb[r] = l_old * scale + sum_exp;

        // Step 4: O_i = O_i * (l_old * scale / l_new) + P_ij * V_j / l_new
        float rescale_o = (l_old * scale) / lUb[r];
        float rescale_p = 1.0f / lUb[r];

        for (uint32_t d = 0; d < D; d++) {
            float pv = 0.0f;
            for (uint32_t c = 0; c < B; c++) {
                pv += pUb[c] * vUb[c * D + d];
            }
            oUb[r * D + d] = oUb[r * D + d] * rescale_o + pv * rescale_p;
        }

        mUb[r] = m_new;
    }
}
```

## 5. 从 naive 到 online_softmax 的关键修改点

| 修改项 | naive（优化前） | online_softmax（优化后） |
|--------|---------------|------------------------|
| 计算范式 | 先完整算 $QK^T$，再 Softmax，再 ×V | 逐 tile 计算，增量更新 running 状态 |
| 内存复杂度 | $O(S^2)$，需完整矩阵 | $O(S)$，仅当前 tile + running m/l/O |
| max/sum 计算 | 双 pass（先 max 再 sum） | **单 pass**，tile 内即时更新 |
| 数值稳定性 | 减 max 需额外遍历 | 天然 Safe Softmax，每步减当前 m_new |
| 输出更新 | 最后一次性 prob × V | 每 tile 增量加权累加 $P_{ij} \cdot V_j$ 到 O |
| 跨 tile 状态 | 无（全量后一次性处理） | **running m、running l、O_acc** 跨 tile 传递 |

## 6. 注意事项 / 约束

1. **数值稳定性：必须维护 running max**。$m_{new} = \max(m_{old}, m_{local})$，所有指数计算以 $m_{new}$ 为基准减。

2. **O_acc 的修正因子**。当 $m_{new} > m_{old}$ 时，$O_{old}$ 和 $l_{old}$ 需乘 $e^{m_{old} - m_{new}}$，scale 必须在更新前计算。

3. **与 GM workspace 的关系**。Flash Attention 通常需要 workspace 保存跨 tile 的 running m/l。本优化是**算法层面**的内存降低，workspace 属于**工程层面**的跨调用状态传递。

4. **last tile 的边界处理**。当 $S$ 不是 $B$ 整数倍时，最后一个 tile 的有效列数小于 $B$，避免 padding 值干扰。

5. **精度与性能的平衡**。$m$ 和 $l$ 建议用 FP32 维护，即使输入/输出是 FP16。

6. **Ascend Cube-Vector 负载均衡**。在 Ascend 910B 上，$QK^T$ 运行在 CUBE 核心，softmax 和 $P \times V$ 运行在 Vector 核心，优化时需兼顾两端。

7. **Workspace MTE3 写回是显著开销**。running 状态跨 S2 tile 传递通常需要 GM workspace，MTE3 向 workspace 的写回是性能瓶颈之一。

8. **Causal Mask 的 Block-level Skip**。Decoder 场景下 causal mask 可在 tile 级别做粗粒度跳过：
   - 若 $s2\_start > s1\_end$：**SKIP** 整个 block
   - 若 $s2\_end \le s1\_start$：**FULL** block
   - 否则：**PARTIAL** block（需 element-level masking）

   理论跳过率：$(n-1)/(2n) \to 50\%$。实际：$S=32K$ 约 45%，$S=2K$ 约 37%，$S=512$ 约 22%。

9. **KV-Cache：Prefill vs Decode 的差异**。
   - **Prefill 阶段**：$Q$ 为完整序列，需完整 tiling。
   - **Decode 阶段**：$Q$ 长度为 1，无需在 $Q$ 维度 tiling，仅沿 KV 序列维度分 tile，完全 memory-bound。

10. **Sliding Window Attention**。若使用滑动窗口（仅 attend 最近 $W$ 个 token），valid 条件为 $i - W < j \le i$。Block-level 可额外跳过 $s2\_end < s1\_start - W$ 的 block。

11. **preTokens / nextTokens 参数**。Ascend Attention API 通过 `preTokens` 和 `nextTokens` 定义 attention 窗口，等效于 band mask。配置时需确保与 causal / sliding window 逻辑一致。

## 7. 选型决策与自检清单

### 7.1 选型决策

```
if (算子包含 softmax && 输入序列长度 S > 512):
    → 启用 online_softmax（ tiled 实现）
    → B = 64 或 128（对齐 Cube 粒度）
    → running 状态用 FP32
else:
    → 标准 Safe Softmax 即可（S 小，全量矩阵 fit SRAM）
```

## 8. AscendC Kernel 优化实现

上述设计在 Python 层逐 tile 实现时，受限于 Python → NPU 的调度开销和独立的 kernel launch，延迟数据不能反映算法真实优势。**生产级优化需在 AscendC 层面将 QK^T、Online Softmax 更新、PV 乘积全部融合到单个 kernel 内。**

### 8.1 从 Python 层到 AscendC 的关键跨越

| 维度 | Python 层实现 | AscendC Fused Kernel |
|------|--------------|---------------------|
| 调度开销 | 16~128 次 Python 循环 + kernel launch | **单次 kernel launch**，零 Python 开销 |
| Score 计算 | `torch.matmul` 独立调用 | 内联 `Mul` + `ReduceSum`（Vector PIPE） |
| Softmax | `torch.max`/`torch.exp`/`torch.sum` | `ReduceMax` → `Adds` → `Exp` → `ReduceSum` |
| P @ V | `torch.matmul` 独立调用 | 内联标量累加（V 为行优先，列访问需 strided） |
| 并行度 | batch 维度串行 | **Multi-block**：每个 AICore 处理一个或多个 Q row |
| Causal Skip | Python `if` 判断 | **Block-level `continue`** 跳过整 tile |

### 8.2 AscendC Kernel 架构

```
每个 AI Core Block 处理一个或多个 Q rows
├─ Load Q row [D] → UB
├─ Cast FP16 → FP32, Muls(scale)
├─ Init: m=-inf, l=0, O_acc=0
├─ For each S2 tile j:
│   ├─ Causal check: tileStart > s1Idx ? skip
│   ├─ MTE2: Load K tile [actualB, D] → UB
│   ├─ Vector: Cast K→FP32, Mul(q, k_slice), ReduceSum → score[actualB]
│   ├─ MTE2: Load V tile [actualB, D] → UB
│   ├─ Causal: mask upper part to -inf (if tile crosses diagonal)
│   ├─ Vector: ReduceMax(score) → m_local
│   ├─ Vector: Adds(score, -m_new), Exp → P[actualB]
│   ├─ Vector: ReduceSum(P) → sum_exp
│   ├─ Scalar: update m, l, rescale factors
│   ├─ Vector: Muls(O_acc, rescale_o)
│   └─ Scalar+Vector: P @ V column-by-column, accumulate to O_acc
├─ Cast O_acc → FP16
├─ MTE3: Write O row → GM
└─ MTE3: Write L = l_final → GM
```

### 8.3 Vector API 替换映射

```cpp
// Score: Q @ K^T (原为标量循环)
Cast(kFloat, kLocal, ..., actualB * D);
for (b = 0; b < actualB; b++) {
    Mul(tmp, qFloat, kFloat[b * D], D);       // Vector 逐元素乘
    ReduceSum(sumBuf, tmp, sumBuf, D);         // Vector 规约求和
    score.SetValue(b, sumBuf.GetValue(0));
}

// Softmax: max → exp → sum (原为标量循环)
ReduceMax(maxBuf, score, maxBuf, actualB);     // Vector 规约求最大
Adds(tmp, score, -mNew, actualB);              // Vector 逐元素加
Exp(tmp, tmp, actualB);                        // Vector 指数
ReduceSum(sumBuf, tmp, sumBuf, actualB);       // Vector 规约求和

// O rescale (原为标量循环)
Muls(oAcc, oAcc, rescaleO, D);                 // Vector 逐元素乘标量
```

### 8.4 多核并行策略

- **总 row 数**：`totalRows = batch × num_heads × S1`
- **Block 分配**：`blockDim = min(totalRows, 8)`（匹配 Ascend 910B 的 8 AICore）
- **每 Block 工作量**：`rowsPerCore = ceil(totalRows / blockDim)`
- **Flat index 分解**：`row → (batchIdx, headIdx, s1Idx)`

### 8.5 UB 内存预算（优化后）

| Buffer | 用途 | D=64, tileB=128 | D=128, tileB=128 |
|--------|------|----------------|-----------------|
| qBuf | Q row (FP16) | 128 B | 256 B |
| kBuf/vBuf | K/V tile (FP16) | 2 × 16 KB | 2 × 32 KB |
| qFloatBuf | Q row (FP32) | 256 B | 512 B |
| kvFloatBuf | K/V float (时间复用) | 32 KB | 64 KB |
| scoreBuf/pBuf | Score / Prob (FP32) | 2 × 512 B | 2 × 512 B |
| oAccBuf | Output accumulator | 256 B | 512 B |
| tmpBuf/reduceBuf | Vector workspace | 2 × 512 B | 2 × 512 B |
| **总计** | | **~100 KB** | **~135 KB** |

UB 容量 192 KB，两种配置均安全。K/V float buffer 时间复用节省 32~64 KB。

### 8.6 预期性能

| 指标 | Python 层 Online | AscendC Fused |
|------|-----------------|---------------|
| S=512 | 慢于 naive 28x | **预计快 1.2~1.5x** |
| S=4K | 慢于 naive 1.4x | **预计快 1.5~3x** |
| S=32K | 慢于 naive 25x | **预计快 2~5x** (causal+skip) |
| 内存 | O(S) | O(S)（与 Python 层相同） |

> **关键结论**：AscendC kernel 消除了 Python 调度开销，使 Online Softmax 的带宽优势（省去 O(S²) 中间矩阵搬运）真正转化为延迟收益。


### 7.2 自检清单

- [ ] running m 初始化为 `-inf`，running l 初始化为 `0`
- [ ] 使用 `m_new = max(m_old, m_local)` 更新
- [ ] $scale = e^{m_{old} - m_{new}}$ 在更新 l 和 O 之前计算
- [ ] O_acc 和 running l 使用 FP32，即使输入输出为 FP16
- [ ] Last tile 边界处理：有效列数 = `min(B, seqLen - j * B)`
- [ ] Causal mask 场景：全 masked tile 正确跳过更新
- [ ] 精度校验通过：与 naive Safe Softmax 对比，误差 < 1e-5（FP32）或 < 1e-3（FP16）
- [ ] AscendC kernel 使用 Vector API 替代标量循环（`GetValue`/`SetValue` 仅在必要处使用）
- [ ] Multi-block 并行正确分配 Q rows，无重复/遗漏
- [ ] UB 内存总量 < 192 KB（Ascend 910B）
