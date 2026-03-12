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

llvm::DenseSet<llvm::StringRef> BishengCluster::getClusterableOps() {
  return llvm::DenseSet<llvm::StringRef>({
    "mfuse.add",
    "mfuse.sub",
    "mfuse.mul",
    "mfuse.div",
  });
}

bool BishengCluster::canClusterableOp(const llvm::DenseSet<llvm::StringRef> &opList, Operation *op) {
  if (op == nullptr) {
    return false;
  }

  StringRef opName = op->getName().getStringRef();

  if (opList.find(opName) == opList.end()) {
    MLOG(DEBUG) << "Op not in Bisheng cluster list: " << opName;
    return false;
  }

  if (op->getNumResults() > 0) {
    Type outputType = op->getResult(0).getType();
    if (mlir::isa<ComplexType>(outputType)) {
      return false;
    }
    if (auto tensorType = dyn_cast<TensorType>(outputType)) {
      if (!tensorType.hasStaticShape()) {
        MLOG(DEBUG) << "Op result has dynamic shape, rejected by Bisheng: " << opName;
        return false;
      }
    }
  }

  for (Value operand : op->getOperands()) {
    if (auto tensorType = dyn_cast<TensorType>(operand.getType())) {
      if (!tensorType.hasStaticShape()) {
        MLOG(DEBUG) << "Op operand has dynamic shape, rejected by Bisheng: " << opName;
        return false;
      }
    }
  }

  if (hasZeroShape(op)) {
    MLOG(DEBUG) << "Op has zero shape: " << opName;
    return false;
  }

  return true;
}

llvm::DenseSet<llvm::StringRef> BishengCluster::getClusterableOpList() { return getClusterableOps(); }

bool BishengCluster::isClusterableOp(Operation *op) { return canClusterableOp(opList_, op); }

std::string BishengCluster::getFusionType() { return "bisheng"; }

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//
#define GEN_PASS_DEF_BISHENGCLUSTER
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

struct BishengClusterPass : public impl::BishengClusterBase<BishengClusterPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    BishengCluster cluster;
    if (cluster.run(funcOp)) {
      MLOG(DEBUG) << "BishengCluster modified function: " << funcOp.getName();
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createBishengClusterPass() { return std::make_unique<mfuse::BishengClusterPass>(); }

}  // namespace mlir
