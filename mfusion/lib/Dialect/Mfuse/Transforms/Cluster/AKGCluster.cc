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

#include "mfusion/Dialect/Mfuse/Transforms/Cluster/BaseCluster.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mfusion/Dialect/Mfuse/Transforms/Cluster/Utils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"
#include "mfusion/Support/Logging.h"

namespace mlir {
namespace mfuse {

// Get clusterable operations specifically for the AKG backend
llvm::DenseSet<llvm::StringRef> AKGCluster::getClusterableOps() {
  return llvm::DenseSet<llvm::StringRef>({
    // Element-wise unary
    "mfuse.abs",
    "mfuse.ceil",
    "mfuse.exp",
    "mfuse.floor",
    "mfuse.log",
    "mfuse.logical_not",
    "mfuse.neg",
    "mfuse.reciprocal",
    "mfuse.rsqrt",
    "mfuse.sqrt",
    "mfuse.trunc",

    // Element-wise binary
    "mfuse.add",
    "mfuse.div",
    "mfuse.eq",
    "mfuse.ge",
    "mfuse.gt",
    "mfuse.le",
    "mfuse.logical_and",
    "mfuse.logical_or",
    "mfuse.lt",
    "mfuse.maximum",
    "mfuse.minimum",
    "mfuse.mul",
    "mfuse.ne",
    "mfuse.pow",
    "mfuse.real_div",
    "mfuse.sub",
  });
}

// Ensure the OP meets the constraints of the AKG backend (e.g., static shape, support constraint)
bool AKGCluster::canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op) {
  if (op == nullptr) {
    return false;
  }

  StringRef opName = op->getName().getStringRef();

  // Check if operation is in clusterable whitelist
  if (opList.find(opName) == opList.end()) {
    MLOG(DEBUG) << "Op not in AKG cluster list: " << opName;
    return false;
  }

  // Check if output type is complex type (Not supported generally in AKG flow)
  if (op->getNumResults() > 0) {
    Type outputType = op->getResult(0).getType();
    if (mlir::isa<ComplexType>(outputType)) {
      return false;
    }
    // Only support static shape for AKG (Polyhedral model constraints)
    if (auto tensorType = dyn_cast<TensorType>(outputType)) {
      if (!tensorType.hasStaticShape()) {
        MLOG(DEBUG) << "Op result has dynamic shape, rejected by AKG: " << opName;
        return false;
      }
    }
  }

  // Ensure all operands have static shapes
  for (Value operand : op->getOperands()) {
    if (auto tensorType = dyn_cast<TensorType>(operand.getType())) {
      if (!tensorType.hasStaticShape()) {
        MLOG(DEBUG) << "Op operand has dynamic shape, rejected by AKG: " << opName;
        return false;
      }
    }
  }

  // Reject zero shape
  if (hasZeroShape(op)) {
    MLOG(DEBUG) << "Op has zero shape: " << opName;
    return false;
  }

  return true;
}

llvm::DenseSet<llvm::StringRef> AKGCluster::getClusterableOpList() { return getClusterableOps(); }

bool AKGCluster::isClusterableOp(Operation *op) { return canClusterableOp(opList_, op); }

std::string AKGCluster::getFusionType() { return "akg"; }

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#define GEN_PASS_DEF_AKGCLUSTER
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

struct AKGClusterPass : public impl::AKGClusterBase<AKGClusterPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    AKGCluster cluster;
    if (cluster.run(funcOp)) {
      MLOG(DEBUG) << "AKGCluster modified function: " << funcOp.getName();
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createAKGClusterPass() { return std::make_unique<mfuse::AKGClusterPass>(); }

}  // namespace mlir
