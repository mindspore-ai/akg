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

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

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
        LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << opName
                                << " is not a tensor type, keeping in cluster\n");
        constantsToCluster.insert(defOp);
        continue;
      }

      // Check if this constant is used at a value-dependent index
      if (constIndices != nullptr && constIndices->count(idx) != 0) {
        LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << opName
                                << " is value-dependent, keeping in cluster\n");
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
      LLVM_DEBUG(llvm::dbgs() << "Constant at index " << idx << " for " << opName
                              << " is single-element finite value, keeping in cluster\n");
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
}  // namespace mfuse
}  // namespace mlir
