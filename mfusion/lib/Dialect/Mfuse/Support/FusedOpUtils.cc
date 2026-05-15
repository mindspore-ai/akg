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

#include "mfusion/Dialect/Mfuse/Support/FusedOpUtils.h"

#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Support/Logging.h"

namespace mlir {
namespace mfuse {

namespace {
/// Check if constant value is finite based on its type
bool isFiniteValue(DenseElementsAttr denseAttr) {
  // Handle dense attributes (tensor constants)
  // Get element type and check accordingly
  Type elementType = denseAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    // For all floating point types, use APFloat to check if finite
    auto apfloatValues = denseAttr.getValues<APFloat>();
    if (!apfloatValues.empty()) {
      APFloat value = *apfloatValues.begin();
      return !value.isNaN() && !value.isInfinity();
    }
  }
  // For non-floating point types or if we can't check, consider finite
  return true;
}

bool checkForGroupedMatmul(const std::string &opName, DenseElementsAttr denseAttr, size_t idx) {
  // Special case: bool type or grouped_matmul's int64 group_list parameter
  Type elementType = denseAttr.getElementType();
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    const size_t kGroupListIndex = 7;
    if (intType.getWidth() == 1 ||
        (opName == "mfuse.grouped_matmul" && idx == kGroupListIndex && intType.getWidth() == 64)) {
      return false;
    }
  }
  return true;
}

// DP state for the prefix [0, end).
struct PartitionState {
  size_t preservedOps{0};   // Number of original cluster ops preserved in the prefix plan.
  size_t segmentCount{0};   // Number of valid fused segments selected in the prefix plan.
  int prevIndex{-1};        // Previous DP position used when reconstructing the chosen plan.
  int segmentStart{-1};     // Start index of the segment ending at the current DP position, or -1 if skipped.
  bool initialized{false};  // Whether this DP state has been reached.
};

struct SliceAnalysis {
  bool analyzed{false};
  bool valid{false};
};

bool isBetterPartition(size_t lhsPreservedOps, size_t lhsSegmentCount, size_t rhsPreservedOps, size_t rhsSegmentCount) {
  if (lhsPreservedOps != rhsPreservedOps) {
    return lhsPreservedOps > rhsPreservedOps;
  }
  return lhsSegmentCount < rhsSegmentCount;
}
}  // namespace

/// Collect constant operations that should be included in the cluster.
/// This function analyzes operands of cluster operations and determines which
/// constant operations should be fused into the cluster (vs extracted as inputs).
llvm::DenseSet<Operation *> collectConstantsToCluster(llvm::ArrayRef<Operation *> clusterOps) {
  // Build a set of cluster operations for fast lookup
  llvm::DenseSet<Operation *> clusterOpSet(clusterOps.begin(), clusterOps.end());

  llvm::DenseSet<Operation *> constantsToCluster;
  const auto &opIndexInfo = getConstInputIndexInfo();

  for (Operation *op : clusterOps) {
    std::string opName = op->getName().getStringRef().str();

    // Get the const input indices for this operation type
    const std::unordered_set<size_t> *constIndices = nullptr;
    auto iter = opIndexInfo.find(opName);
    if (iter != opIndexInfo.end()) {
      constIndices = &iter->second;
    }

    for (size_t idx = 0; idx < op->getNumOperands(); ++idx) {
      Value operand = op->getOperand(idx);
      Operation *defOp = operand.getDefiningOp();
      if (defOp == nullptr || clusterOpSet.contains(defOp) || constantsToCluster.contains(defOp)) {
        continue;
      }

      // Check if this is a constant operation
      auto constOp = dyn_cast<mfuse::ConstantOp>(defOp);
      if (!constOp) {
        continue;
      }
      Type resultType = constOp.getResult().getType();

      // If constant is not a tensor type (e.g., scalar), include it in cluster
      if (!isa<TensorType>(resultType)) {
        MLOG(DEBUG) << "Constant at index " << idx << " for " << opName << " is not a tensor type, keeping in cluster";
        constantsToCluster.insert(defOp);
        continue;
      }

      // Check if this constant is used at a value-dependent index
      if (constIndices != nullptr && constIndices->count(idx) != 0) {
        MLOG(DEBUG) << "Constant at index " << idx << " for " << opName << " is value-dependent, keeping in cluster";
        constantsToCluster.insert(defOp);
        continue;
      }

      // Get the constant attribute
      Attribute valueAttr = constOp.getValueAttr();
      if (!valueAttr) {
        continue;
      }

      // Check if it's a single-element finite tensor constant
      auto denseAttr = dyn_cast<DenseElementsAttr>(valueAttr);
      if (!denseAttr || denseAttr.getNumElements() != 1 || !isFiniteValue(denseAttr)) {
        continue;
      }

      if (!checkForGroupedMatmul(opName, denseAttr, idx)) {
        continue;
      }

      // Include it in the cluster
      MLOG(DEBUG) << "Constant at index " << idx << " for " << opName
                  << " is single-element finite value, keeping in cluster";
      constantsToCluster.insert(defOp);
    }
  }

  return constantsToCluster;
}

/// Find external inputs (values defined outside the cluster).
llvm::SetVector<Value> findExternalInputs(llvm::ArrayRef<Operation *> clusterOps,
                                          const llvm::DenseSet<Operation *> &clusterOpSet) {
  llvm::SetVector<Value> externalInputs;
  for (Operation *op : clusterOps) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      // External if defined by block argument or by an op outside the cluster
      if (defOp == nullptr || !clusterOpSet.contains(defOp)) {
        externalInputs.insert(operand);
      }
    }
  }
  return externalInputs;
}

/// Find external outputs (values used outside the cluster).
llvm::SetVector<Value> findExternalOutputs(llvm::ArrayRef<Operation *> clusterOps,
                                           const llvm::DenseSet<Operation *> &constantsToCluster,
                                           const llvm::DenseSet<Operation *> &clusterOpSet) {
  llvm::SetVector<Value> externalOutputs;
  for (Operation *op : clusterOps) {
    if (constantsToCluster.contains(op)) {
      continue;
    }
    for (Value result : op->getResults()) {
      for (Operation *user : result.getUsers()) {
        if (!clusterOpSet.contains(user)) {
          externalOutputs.insert(result);
          break;
        }
      }
    }
  }
  return externalOutputs;
}

Operation *findValidInsertPoint(const llvm::SmallVector<Operation *> &clusterOps,
                                const llvm::SetVector<Value> &externalInputs,
                                const llvm::SetVector<Value> &externalOutputs,
                                const llvm::DenseSet<Operation *> &clusterOpSet) {
  Operation *insertPoint = clusterOps.front();
  for (Value input : externalInputs) {
    Operation *defOp = input.getDefiningOp();
    if (defOp && defOp->getBlock() == insertPoint->getBlock() && insertPoint->isBeforeInBlock(defOp)) {
      insertPoint = defOp;
    }
  }

  for (Value output : externalOutputs) {
    for (Operation *user : output.getUsers()) {
      if (clusterOpSet.contains(user)) {
        continue;
      }
      if (user->getBlock() == insertPoint->getBlock() && !insertPoint->isBeforeInBlock(user)) {
        MLOG(DEBUG) << "FusedOp insert point " << insertPoint->getLoc() << " is not before non-cluster user "
                    << user->getLoc();
        return nullptr;
      }
    }
  }

  return insertPoint;
}

bool buildCluster(llvm::ArrayRef<Operation *> baseClusterOps, ClusterBuildInfo &buildInfo) {
  if (baseClusterOps.size() < kMinClusterSize) {
    return false;
  }

  buildInfo = ClusterBuildInfo{};
  buildInfo.clusterOps.assign(baseClusterOps.begin(), baseClusterOps.end());
  // Re-run constant collection per slice so each partition can fuse the
  // constants it still depends on after splitting.
  buildInfo.constantsToCluster = collectConstantsToCluster(buildInfo.clusterOps);
  std::copy(buildInfo.constantsToCluster.begin(), buildInfo.constantsToCluster.end(),
            std::back_inserter(buildInfo.clusterOps));

  std::sort(buildInfo.clusterOps.begin(), buildInfo.clusterOps.end(),
            [](Operation *lhs, Operation *rhs) { return lhs->isBeforeInBlock(rhs); });

  buildInfo.clusterOpSet.insert(buildInfo.clusterOps.begin(), buildInfo.clusterOps.end());
  buildInfo.externalInputs = findExternalInputs(buildInfo.clusterOps, buildInfo.clusterOpSet);
  buildInfo.externalOutputs =
    findExternalOutputs(buildInfo.clusterOps, buildInfo.constantsToCluster, buildInfo.clusterOpSet);
  if (buildInfo.externalOutputs.empty()) {
    return false;
  }

  buildInfo.insertPoint = findValidInsertPoint(buildInfo.clusterOps, buildInfo.externalInputs,
                                               buildInfo.externalOutputs, buildInfo.clusterOpSet);
  return buildInfo.insertPoint != nullptr;
}

llvm::SmallVector<llvm::SmallVector<Operation *>> partitionClusterOps(llvm::ArrayRef<Operation *> baseClusterOps,
                                                                      size_t minClusterSize) {
  llvm::SmallVector<llvm::SmallVector<Operation *>> partitions;
  const size_t clusterSize = baseClusterOps.size();
  if (clusterSize < minClusterSize) {
    return partitions;
  }

  std::vector<std::vector<SliceAnalysis>> sliceCache(clusterSize, std::vector<SliceAnalysis>(clusterSize + 1));
  auto getOrAnalyze = [&](size_t start, size_t end) -> SliceAnalysis & {
    SliceAnalysis &analysis = sliceCache[start][end];
    if (!analysis.analyzed) {
      analysis.analyzed = true;
      // Cache only legality here. The concrete build info for a slice may
      // become stale after earlier partitions rewrite and erase operations.
      ClusterBuildInfo buildInfo;
      analysis.valid = buildCluster(baseClusterOps.slice(start, end - start), buildInfo);
    }
    return analysis;
  };

  std::vector<PartitionState> dp(clusterSize + 1);
  dp[0].initialized = true;
  for (size_t end = 1; end <= clusterSize; ++end) {
    // Skip the current op and inherit the best plan for the prefix.
    if (dp[end - 1].initialized) {
      dp[end] = dp[end - 1];
      dp[end].prevIndex = static_cast<int>(end - 1);
      dp[end].segmentStart = -1;
      dp[end].initialized = true;
    }

    for (size_t start = 0; start + minClusterSize <= end; ++start) {
      SliceAnalysis &analysis = getOrAnalyze(start, end);
      if (!analysis.valid || !dp[start].initialized) {
        continue;
      }

      // Prefer the plan that preserves more original cluster ops, and break
      // ties by using fewer fused segments.
      size_t candidatePreservedOps = dp[start].preservedOps + (end - start);
      size_t candidateSegmentCount = dp[start].segmentCount + 1;
      if (isBetterPartition(candidatePreservedOps, candidateSegmentCount, dp[end].preservedOps, dp[end].segmentCount)) {
        dp[end].preservedOps = candidatePreservedOps;
        dp[end].segmentCount = candidateSegmentCount;
        dp[end].prevIndex = static_cast<int>(start);
        dp[end].segmentStart = static_cast<int>(start);
        dp[end].initialized = true;
      }
    }
  }

  if (!dp[clusterSize].initialized || dp[clusterSize].preservedOps < minClusterSize) {
    return partitions;
  }

  // Reconstruct the chosen valid slices from the DP predecessor chain.
  for (int end = static_cast<int>(clusterSize); end > 0;) {
    const PartitionState &state = dp[end];
    if (state.segmentStart >= 0) {
      size_t start = static_cast<size_t>(state.segmentStart);
      partitions.emplace_back(baseClusterOps.slice(start, static_cast<size_t>(end) - start).begin(),
                              baseClusterOps.slice(start, static_cast<size_t>(end) - start).end());
    }
    end = state.prevIndex;
  }
  std::reverse(partitions.begin(), partitions.end());
  return partitions;
}
}  // namespace mfuse
}  // namespace mlir
