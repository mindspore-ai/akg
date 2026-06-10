---
name: catlass-api-basics
description: "CATLASS 五层 API 与 Gemm 组装范式：Device/Kernel/Block/Tile、GemmType/GemmShape/DispatchPolicy、标准头文件与 Device 调用流程。适用于 autoresearch 任务中修改 catlass_op 内 .asc/.h 的类型别名区。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc_catlass
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# CATLASS API 基础

CATLASS 是昇腾上的矩阵类算子模板库。AR 任务里，**真正改模板类型别名的位置**通常是：

- `catlass_op/kernel/catlass_kernel.asc` — kernel 侧类型与 `using` 别名
- `catlass_op/include/catlass_kernel.h` — 与 `.asc` 一致的声明

`kernel.py` 里的 `ModelNew` 只负责 `load_library` 和 `torch.ops.catlass.*` 调用，**不要**在 Python 里重写计算逻辑。

典型 `task.yaml` 路径与可编辑列表：

```yaml
catlass:
  root: /path/to/catlass          # CATLASS 仓库根（也可用环境变量 CATLASS_ROOT）
  op_dir: catlass_op              # 相对 task_dir 的 pybind 工程目录（文件夹）
editable_files:
  - kernel.py
  - catlass_op/kernel/catlass_kernel.asc
  - catlass_op/include/catlass_kernel.h
  - catlass_op/src/catlass_torch.cpp
  - catlass_op/CMakeLists.txt
```

`catlass_torch.cpp` 负责 TORCH_LIBRARY 注册、tensor 预处理与 launch；卷积等算子若 profile 显示 Transdata 占比高，往往需改此文件而非仅改 `.asc`。

## 五层架构（从高到低）

| 层级 | 典型入口 | 职责 |
|------|----------|------|
| Device | `Gemm::Device::DeviceGemm<Kernel>` | Host 入口、参数校验、launch |
| Kernel | `BasicMatmul` / `MatmulEpilogue` 等 | Block + 分核 + 同步 |
| Block | `BlockMmad` | 单核 K 维主循环（MMAD + 双缓冲） |
| Tile | `TileCopy` / `TileMmad` | L1/L0 搬运与微内核 |
| Basic | `AscendC::Mmad` / `DataCopy` | 指令级封装 |

## 标准 Gemm 组装顺序

```cpp
// 1. BlockMmad
using ArchTag = Arch::AtlasA2;  // 与目标 NPU 代际一致
using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
using L1TileShape = GemmShape<128, 256, 256>;
using L0TileShape = GemmShape<128, 256, 64>;
using AType = Gemm::GemmType<ElementA, LayoutA>;
using BType = Gemm::GemmType<ElementB, LayoutB>;
using CType = Gemm::GemmType<ElementC, LayoutC>;
using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

// 2. 尾处理（无则 void）
using BlockEpilogue = void;

// 3. 分核 Swizzle
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// 4. Kernel
using MatmulKernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;

// 5. Device
using Matmul = Gemm::Device::DeviceGemm<MatmulKernel>;
```

迁移到 PyTorch 时，**include 列表应与源 example 一致**，不要凭猜测增删 `catlass/...` 头文件。

## 常用头文件（按功能选，勿全抄）

```cpp
#include "catlass/gemm/kernel/basic_matmul.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/arch/arch.hpp"
```

有 Epilogue 时再追加 `matmul_epilogue.hpp`、`epilogue/block/...` 等（见 `catlass-epilogue-composition`）。

## 核心类型

### GemmShape

```cpp
using L1TileShape = GemmShape<M, N, K>;
using L0TileShape = GemmShape<M, N, K>;
```

- `M/N/K` 通常需为 **16 的倍数**
- 常见习惯：`L0.M == L1.M`，`L0.N == L1.N`，`L0.K == L1.K / 4`（非硬性唯一解，但利于调参）

### GemmType

```cpp
using AType = Gemm::GemmType<ElementA, LayoutA>;
```

元素类型与 `Layout`（`RowMajor` / `ColumnMajor` 等）共同决定搬运与 MMAD 行为。

### DispatchPolicy（策略层）

| 策略 | 特点 | 适用 |
|------|------|------|
| `MmadAtlasA2Pingpong<unitFlag>` | L1 双缓冲 | 通用基线 |
| `MmadAtlasA2Preload<unitFlag, shuffleK>` | 预取 + shuffleK | 大 shape、带宽敏感 |
| TLA 系 Pingpong | TLA 模型 | 与 TLA Block 搭配 |

### Layout 与 Swizzle 方向（经验）

- `RowMajor + ColumnMajor`：常见，常配 `L1TileShape<128,256,256>` 一类
- 双 `RowMajor`：`M>N` 与 `M<N` 时 Swizzle 方向可能不同（见 matmul 调优 skill）

## Device 调用骨架

```cpp
Matmul matmulOp;
typename MatmulKernel::Arguments args{problemShape, deviceA, deviceB, deviceC};
matmulOp.CanImplement(args);
size_t wsSize = matmulOp.GetWorkspaceSize(args);
matmulOp.Initialize(args, deviceWorkspace);
matmulOp(stream, aicCoreNum);
```

需要 workspace 的 kernel（如部分 Split-K）必须在 host 侧按 `GetWorkspaceSize` 分配，见迁移规范中的 workspace 规则。

## AR 环境提醒

- 编译依赖 `ASCEND_HOME_PATH`；`CATLASS_ROOT` 来自 `task.yaml catlass.root` 或环境变量；verify 时 `copytree` 整个 `catlass_op/` 文件夹，cmake 传 `-DCATLASS_ROOT=`、`-DNPU_ARCH=`、`-DCATLASS_ARCH=`
- AR profile 与 **triton_ascend** 同路径：`profiler_npu` → `op_statistic.csv` → `generation_profile_result.json`（非 msprof CLI）
- 改 `.asc` 后每轮 eval 会在 verify 目录内 **重新 cmake && make**，以静态断言和链接结果为准
