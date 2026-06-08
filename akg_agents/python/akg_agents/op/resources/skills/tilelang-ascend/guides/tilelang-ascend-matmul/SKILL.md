---
name: tilelang-ascend-matmul
description: "TileLang Ascend 矩阵乘法算子编码指南。涵盖 Expert/Developer 两种模式的 GEMM 编码范式差异、K 维循环累加、带转置的 GEMM、GEMM 非整除维度处理。当生成 matmul / GEMM / 线性层类算子时参考此指南。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: tilelang_ascend
  hardware: "Atlas A2, Atlas A3"
  operator_type: "matmul"
---

# TileLang Ascend 矩阵乘法编码指南

---

## 决策树：matmul 算子路径

**重要**：`T.reduce_sum/max/min` 和 `T.tile.*` 在 Developer 和 Expert 模式下**都可使用**。模式选择取决于是否需要手动控制内存层级和同步，而非使用了哪个 API。

```
含 matmul / @ / 矩阵乘
├─ 仅 matmul → 纯 Cube
│   模式: Expert（手动管理 L0）
│   API（Ascend 专用）: T.gemm_v0(A_L1, B_L1, C_L0C, transpose_A, init)
│   内存（Expert）: T.alloc_L1 → T.alloc_L0C
│   内存（Developer）: T.alloc_shared → T.alloc_fragment
│   同步: T.barrier_all() + T.Scope("C")
│   Kernel: T.Kernel(一维, is_npu=True) as (cid, _)
│
└─ matmul + element-wise 后处理 → 混合（融合算子）
    模式: Developer + 自动同步（推荐）或 Expert + 手动同步
    API: T.gemm_v0 + T.tile.* / T.Parallel + workspace
    内存: GM→L1→L0A/L0B→L0C→workspace→UB→GM
    workspace: 数量/shape/dtype 自动推断，位于 GM
    pass_configs: AUTO_CV_COMBINE:True + AUTO_CV_SYNC:True + AUTO_SYNC:True
    同步: 自动（AUTO_CV_SYNC）或手动（T.set_cross_flag / T.wait_cross_flag）
```

