/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/RemoveRedundantLoops.h"
#include <optional>
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Transforms/RegionUtils.h"
#include "akg/Utils/Constants.h"

namespace mlir {
#ifndef GEN_PASS_DEF_REMOVEREDUNDANTLOOPS
#define GEN_PASS_DEF_REMOVEREDUNDANTLOOPS
#ifndef GEN_PASS_DECL_REMOVEREDUNDANTLOOPS
#define GEN_PASS_DECL_REMOVEREDUNDANTLOOPS
#include "akg/Dialect/Affine/Passes.h.inc"

#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "remove-redundant-loops"

namespace {
using namespace mlir;  // NOLINT(build/namespaces)
using namespace llvm;  // NOLINT(build/namespaces)

// If the value of tripcount in the for loop is less than or equal to 1, the for loop runs only once.
// This indicates that the for loop is redundant and needs to be deleted.
//
class RemoveRedundantLoops : public impl::RemoveRedundantLoopsBase<RemoveRedundantLoops> {
 public:
  RemoveRedundantLoops() {}
  void runOnOperation() override;
};

}  // namespace

void RemoveRedundantLoops::runOnOperation() {
  func::FuncOp func = getOperation();
  OpBuilder b(func);
  SmallVector<affine::AffineForOp, kSmallVectorSizeEight> redundantLoops;
  func->walk([&redundantLoops](affine::AffineForOp forOp) {
    // skipped for now
    // %0 = affine.for %arg2 = 0 to 4 step 4 iter_args(%arg3 = %cst_0) -> (vector<4xf32>)
    if (!forOp.getResults().empty()) {
      return;
    }

    std::optional<uint64_t> maybeConstTripCount = getConstantTripCount(forOp);
    if (!maybeConstTripCount) {
      return;
    }

    if (maybeConstTripCount <= 1) {
      redundantLoops.push_back(forOp);
    }
  });

  for (auto forOp : redundantLoops) {
    affine::AffineForOp dependentOp = nullptr;
    for (auto value : forOp.getOperation()->getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (isa<IndexType>(blockArg.getType())) {
          Block *block = blockArg.getOwner();
          Operation *parentOp = block->getParentOp();
          if ((parentOp != nullptr) && isa<affine::AffineForOp>(parentOp)) {
            dependentOp = dyn_cast<affine::AffineForOp>(parentOp);
          }
        }
      }
    }

    b.setInsertionPoint(forOp);
    IRMapping map;
    if (dependentOp) {
      // exist dependent loop
      map.map(forOp.getInductionVar(), dependentOp.getInductionVar());
    } else {
      // the upper and lower bounds are constants.
      auto lbMap = forOp.getLowerBoundMap();
      if (lbMap.isSingleConstant()) {
        mlir::Value lbConstOp = b.create<arith::ConstantIndexOp>(forOp.getLoc(), lbMap.getSingleConstantResult());
        map.map(forOp.getInductionVar(), lbConstOp);
      }
    }
    for (auto &op : forOp.getBody()->without_terminator()) {
      (void)b.clone(op, map);
    }
    forOp.erase();
  }
}

namespace mlir {
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveRedundantLoopsPass() {
  return std::make_unique<RemoveRedundantLoops>();
}
}  // namespace mlir
