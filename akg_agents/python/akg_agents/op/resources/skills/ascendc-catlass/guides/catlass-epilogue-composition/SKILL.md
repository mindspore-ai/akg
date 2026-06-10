---
name: catlass-epilogue-composition
description: "CATLASS 尾处理：MatmulEpilogue、EVG TreeVisitor、UB workspace；Add/Bias/ReLU 等融合与对应 Kernel 选型。适用于 matmul+逐元素类融合算子。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "matmul, fused"
---

# CATLASS 尾处理（Epilogue）

reference 若是 `D = matmul(A,B) (+/-) 逐元素运算`，需要在 Gemm 上挂 **BlockEpilogue** 或 **EVG**，而不是在 `kernel.py` 里用 `F.relu(matmul(...))` 代替（会偏离 catlass 优化目标，且可能触发静态检查）。

## 路径对比

| 路径 | Kernel | 说明 |
|------|--------|------|
| 无尾处理 | `BasicMatmul<..., void, Scheduler>` | 纯 GEMM |
| 标准 Epilogue | `MatmulEpilogue` | AIC 内 MMAD + 逐元素 |
| EVG + GM | `BasicMatmulTlaVisitor` | MMAD 结果经 workspace，AIV 做树形融合 |
| EVG + UB | `BasicMatmulTlaUbVisitor` | 累加留 UB，减少 GM 往返 |

## 标准 Epilogue（以 Add 为例）

额外头文件示例：

```cpp
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
```

组装要点：

```cpp
using BlockMmad = /* 同 BasicMatmul */;

using XType = Gemm::GemmType<half, layout::RowMajor>;  // 与融合语义一致
using DType = CType;
using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2ElemWiseOneSource;
constexpr uint32_t computeLength = 16384;  // 按 tile 与架构调整
using TileElemWise = Epilogue::Tile::TileElemWiseAdd<ArchTag, CType, computeLength>;
using EpilogueTileCopy = Epilogue::Tile::TileCopy<ArchTag, CType, XType, DType>;
using BlockEpilogue = Epilogue::Block::BlockEpilogue<
    EpilogueDispatchPolicy, CType, XType, DType, TileElemWise, EpilogueTileCopy>;

using MatmulKernel = Gemm::Kernel::MatmulEpilogue<BlockMmad, BlockEpilogue, BlockScheduler>;
```

`MatmulEpilogue` 参数语义上包含 **偏置/加数矩阵 X** 与输出 D；与 `03_matmul_add` 一类 example 对齐。

## 自定义逐元素（如 Add+ReLU）

仓内若无现成 `TileElemWise*`：

- 可参照 example 增加 **自定义 Tile Epilogue** 头文件，并在 `.asc` 中挂到 `BlockEpilogue`
- AR 任务若未把该头文件列入 `editable_files`，则只能在已有 Tile 组合内选型

## EVG（TreeVisitor）概要

适合更复杂的融合图（多输入、多算子链）：

```cpp
#include "catlass/gemm/kernel/basic_matmul_tla_visitor.hpp"
#include "catlass/epilogue/block/block_epilogue_visitor.hpp"
#include "catlass/epilogue/fusion/tree_visitor.hpp"
```

用 `TreeVisitor` 组合 `VisitorAccLoad`、`VisitorAuxLoad`、`VisitorCompute<Op>`、`VisitorAuxStore` 等；`Arguments` 需额外携带 `EVG::Arguments`。

UB 版本仅将 `EpilogueVisitor<false>` 换为 `<true>`，Kernel 换为 `BasicMatmulTlaUbVisitor`。

## 融合模式速查

| 目标 | 组件方向 |
|------|----------|
| D = C + X | `TileElemWiseAdd` / `VisitorCompute<Add>` |
| 仅 GEMM | `BlockEpilogue = void` |
| ReLU / GELU / SiLU | EVG `VisitorCompute<Relu>` 等 |
| 带 bias 的 GEMM | 部分模板在 `BlockMmad` 上增加 Bias 类型参数 |

## Kernel 选型

| 需求 | Kernel |
|------|--------|
| 无融合 | `BasicMatmul` |
| 简单逐元素、AIC 内完成 | `MatmulEpilogue` |
| 复杂融合、AIV 执行 | `BasicMatmulTlaVisitor` |
| 复杂融合 + 少 GM 往返 | `BasicMatmulTlaUbVisitor` |

## AR 与 Python 对齐

- `reference.py` 的 `Model.forward` 定义了语义（如 `relu(A@B+X)`）
- `kernel.py` 的 `ModelNew` 应调用 **同一语义** 的 `torch.ops.catlass.*`
- 融合逻辑应在 **catlass 侧** 实现（`.asc` / 自定义 epilogue 头文件 / `catlass_torch.cpp`）；不要在 Python 里用纯 torch 重算一遍再冒充 catlass 结果
