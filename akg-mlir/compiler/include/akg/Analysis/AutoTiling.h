/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMPILER_INCLUDE_AKG_ANALYSIS_AUTOTILING_H_
#define COMPILER_INCLUDE_AKG_ANALYSIS_AUTOTILING_H_
#include <vector>
#include "akg/Analysis/Model.h"
#include "akg/Analysis/TilingSolver.h"
#include "akg/Analysis/TilingStrategy.h"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace autotiling {
using llvm::SmallVector;
/// That is a data structure that wrap the info for tiling task.
struct TilingTaskDesc {
  TilingTaskDesc(size_t b, size_t l) : bandIdx(b), level(l) {}
  size_t bandIdx;
  size_t level;
};

// Affine version
InitGraphPtr parseIr(Operation *funcOp, const std::vector<SmallVector<affine::AffineForOp, 6>> &bands);
InitGraphPtr parseIr(const std::vector<SmallVector<affine::AffineForOp, 6>> &bands);

// SCF version
InitGraphPtr parseIr(Operation *funcOp, const std::vector<SmallVector<scf::ForOp, 6>> &bands);
InitGraphPtr parseIr(const std::vector<SmallVector<scf::ForOp, 6>> &bands);

/// Analyze some contextual information (such as whether subsequent passes need to be opened double_buffer), as well as
/// some preliminary strategies (such as operator priority, memory bottleneck statements)
ModelGraphPtr buildModelGraph(InitGraphPtr initGraph);

/// Because these preliminary analyses are hardware-specific, ModelGraph subclass inheritance for different
/// hardware platforms can be (e.g., GpuModelGraph, CpuModelGraph and NpuModelGraph).
GpuModelGraphPtr buildGpuModelGraph(InitGraphPtr initGraph, const TilingStrategyManagerPtr tilingMgr);
NpuModelGraphPtr buildNpuModelGraph(InitGraphPtr initGraph, const TilingStrategyManagerPtr tilingMgr);
CpuModelGraphPtr buildCpuModelGraph(InitGraphPtr initGraph, const TilingStrategyManagerPtr tilingMgr);

/// Encapsulates the steps required to solve sharding, and supports two solution methods: heuristic and search (Tuning);
TilingSolverPtr getHeuristicTilingSolver(ModelGraphPtr modelGraph);

// Affine version
void getTileSizeWithSolver(const TilingSolverPtr &solver, SmallVector<affine::AffineForOp, 6> band,
                           SmallVectorImpl<unsigned> *tileSizes, const TilingTaskDesc &taskDesc);

// SCF version
void getTileSizeWithSolver(const TilingSolverPtr &solver, SmallVector<scf::ForOp, 6> band,
                           SmallVectorImpl<unsigned> *tileSizes, const TilingTaskDesc &taskDesc);
}  // namespace autotiling
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_ANALYSIS_AUTOTILING_H_
