---
name: tilelang-ascend-attention
description: "TileLang Ascend Attention 算子编码指南。涵盖 Cube+Vector 融合编程范式、workspace 跨核通信、online softmax 累加模式。当生成 Attention 类算子时参考此指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "attention"
---

# TileLang Ascend Attention 算子编码指南

---

## Cube+Vector 融合编程范式

Attention 是典型的 Cube+Vector 融合场景：QK^T 用 Cube 核（GEMM），softmax/reduce 用 Vector 核，PV 用 Cube 核（GEMM）。

**数据流**：Cube 和 Vector 通过 **workspace** 通信，形成 Cube 写 → Vector 读 → Vector 写 → Cube 读的循环。

**关键要点**：
- `workspace_idx=[4, 5, 6, 7, 8]` 在 `@tilelang.jit` 中声明 workspace tensor
- Developer 模式下开启 `AUTO_CV_COMBINE + AUTO_CV_SYNC` 自动处理 C/V 融合与同步
- Cube 域内必须有 `T.barrier_all()` 同步 C/V 核

---

## workspace 跨核通信

workspace 在 `@tilelang.jit` 中通过 `workspace_idx` 声明，用于 Cube 核和 Vector 核之间传递中间结果。

**典型 workspace 用途**（以 sparse_flash_attention 为例）：
- `workspace_1/2`：KV 数据，Vector 核写入 → Cube 核读取
- `workspace_3`：QK^T 结果（acc_s），Cube 核写入 → Vector 核读取
- `workspace_4`：softmax 后的注意力权重，Vector 核写入 → Cube 核读取
- `workspace_5`：PV 结果（acc_o），Cube 核写入 → Vector 核读取

**必需的 pass_configs**：

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

---

## Online Softmax 累加模式

Attention 的 softmax 需要分块计算（KV 序列太长无法一次放入 UB），使用 online softmax 算法维护三个 running state：`m_i`（行最大值）、`sumexp`（行指数和）、`acc_o`（输出累加）。

**算法流程**：

```
初始化: m_i = -inf, sumexp = 0, acc_o = 0

对每个 KV 块:
  1. 从 workspace 读取 Cube 计算的 acc_s
  2. m_i_new = max(m_i_prev, max(acc_s))
  3. correction = exp(m_i_prev - m_i_new)
  4. sumexp = sumexp * correction + sum(exp(acc_s - m_i_new))
  5. acc_o = acc_o * correction + acc_s_normalized @ V_partial
```

**correction 因子分解**：代码中将 `exp(m_prev - m_new)` 拆为三步：
1. `m_i_prev[i] = m_i_prev[i] - m_i[i]` → `m_i_prev[i] = T.exp(m_i_prev[i])` 计算 correction
2. `sumexp[i] *= m_i_prev[i]` 修正历史 sumexp
3. `acc_o[h_i, j] = acc_o[h_i, j] * m_i_prev[h_i]` 修正历史输出

---

## 编码要点

1. **L0C 不能直接做 reduce**：必须 `T.copy` 到 UB 后再做 softmax 归约
2. **init 参数**：首块 `init=True` 清零 L0C，后续块 `init=False` 累加
3. **correction 因子**：online softmax 中 `exp(m_prev - m_new)` 是数值稳定的关键，不可省略
4. **SEQ_LEN 非对齐**：用 `T.ceildiv` 处理，不要假设整除
5. **Developer 模式优先**：除非需要极致性能，否则用 `pass_configs` + 自动同步，避免手写 `T.set_flag`/`T.wait_flag`
6. **workspace 索引**：使用 `cid`（kernel id）索引 workspace 的第一维，确保不同核写入不同区域
7. **v_block 切分**：Vector 核按 `vid` 分割处理 L0C 的行，每个 Vector 核处理 `v_block = H_per_block // 2` 行
