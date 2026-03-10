/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MFUSION_DIALECT_MFUSE_UTILS_FUSED_OP_UTILS_H
#define MFUSION_DIALECT_MFUSE_UTILS_FUSED_OP_UTILS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace mfuse {
/// Collect constant operations that should be included in the cluster.
/// This function analyzes operands of cluster operations and determines which
/// constant operations should be fused into the cluster (vs extracted as inputs).
llvm::DenseSet<Operation *> collectConstantsToCluster(llvm::ArrayRef<Operation *> clusterOps);

llvm::SetVector<Value> findExternalInputs(llvm::ArrayRef<Operation *> clusterOps,
                                          const llvm::DenseSet<Operation *> &clusterOpSet);

/// Find external outputs (values used outside the cluster).
llvm::SetVector<Value> findExternalOutputs(llvm::ArrayRef<Operation *> clusterOps,
                                           const llvm::DenseSet<Operation *> &constantsToCluster,
                                           const llvm::DenseSet<Operation *> &clusterOpSet);

/// Find a valid insertion point for the fused operation.
/// The insertion point must satisfy two SSA dominance constraints:
///   1. After all external input definitions (so they dominate the fused op's uses).
///   2. Before all non-cluster users of external outputs (so fused results dominate users).
/// Returns nullptr if no valid insertion point exists.
Operation *findValidInsertPoint(const llvm::SmallVector<Operation *> &clusterOps,
                                const llvm::SetVector<Value> &externalInputs,
                                const llvm::SetVector<Value> &externalOutputs,
                                const llvm::DenseSet<Operation *> &clusterOpSet);
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_UTILS_FUSED_OP_UTILS_H
