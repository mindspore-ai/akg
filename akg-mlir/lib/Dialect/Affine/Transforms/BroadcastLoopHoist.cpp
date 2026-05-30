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

#include "akg/Dialect/Affine/Transforms/BroadcastLoopHoist.h"
#include "akg/Dialect/Affine/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_BROADCASTLOOPHOIST
#define GEN_PASS_DECL_BROADCASTLOOPHOIST
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "broadcast-loop-hoist"

namespace mlir {
namespace {

struct BroadcastLoopHoist : public impl::BroadcastLoopHoistBase<BroadcastLoopHoist> {
  BroadcastLoopHoist() {}

  void runOnOperation() override;

 private:
  bool isBroadcastLoop(affine::AffineForOp forOp);
  bool isInnermostForOp(affine::AffineForOp forOp);
  bool hoistBroadcastLoops(func::FuncOp funcOp);

  /// Interchange outerLoop and innerLoop. The innerLoop is hoisted to become the
  /// new outer loop. Returns the new outer loop on success so that the caller
  /// can continue hoisting the same logical loop further upward.
  FailureOr<affine::AffineForOp> interchangeLoops(affine::AffineForOp outerLoop, affine::AffineForOp innerLoop);
};

bool BroadcastLoopHoist::isBroadcastLoop(affine::AffineForOp forOp) { return forOp->hasAttr(kBroadcastLoopAttr); }

/// Returns true if no other affine.for ops are nested within this op.
bool BroadcastLoopHoist::isInnermostForOp(affine::AffineForOp forOp) {
  bool innermost = true;
  forOp.walk([&](affine::AffineForOp nested) {
    if (nested != forOp) {
      innermost = false;
    }
  });
  return innermost;
}

/// Check that the inner loop is the last non-terminator op in the outer loop's body.
/// Ops before the inner loop are allowed (non-perfect nest), but ops after it are not,
/// because after interchange those ops' execution count would change.
static bool canInterchange(affine::AffineForOp outerLoop, affine::AffineForOp innerLoop) {
  auto &ops = outerLoop.getBody()->getOperations();
  for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
    if (it->hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    // The first non-terminator op from the end must be the inner loop.
    if (&*it != innerLoop.getOperation()) {
      LLVM_DEBUG(llvm::dbgs() << "Inner loop is not the last op in outer loop body, "
                              << "cannot safely interchange\n");
      return false;
    }
    return true;
  }
  return false;
}

/// Determine how many levels the broadcast loop should be hoisted based on
/// the index order in the store operation within the loop. The broadcast loop's
/// IV position in the store indices defines where it should sit in the loop
/// nest. We count how many other loop IVs appear after it — that is how many
/// levels to hoist.
static int getNumLevelsToHoist(affine::AffineForOp broadcastLoop) {
  Value broadcastIV = broadcastLoop.getInductionVar();

  // Find a store operation within the broadcast loop.
  affine::AffineStoreOp targetStore;
  broadcastLoop.walk([&](affine::AffineStoreOp storeOp) { targetStore = storeOp; });

  if (!targetStore) {
    return 0;
  }

  // Find the position of the broadcast IV in the store's map operands.
  auto mapOperands = targetStore.getMapOperands();
  int broadcastPos = -1;
  for (int i = 0; i < static_cast<int>(mapOperands.size()); ++i) {
    if (mapOperands[i] == broadcastIV) {
      broadcastPos = i;
      break;
    }
  }

  if (broadcastPos < 0) {
    return 0;
  }

  // Count how many affine.for loop IVs appear after the broadcast IV
  // in the store indices. These loops should be inner to the broadcast loop.
  int numAfter = 0;
  for (int i = broadcastPos + 1; i < static_cast<int>(mapOperands.size()); ++i) {
    if (auto blockArg = dyn_cast<BlockArgument>(mapOperands[i])) {
      if (isa<affine::AffineForOp>(blockArg.getOwner()->getParentOp())) {
        ++numAfter;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Broadcast IV at store position " << broadcastPos << ", need to hoist " << numAfter
                          << " levels\n");
  return numAfter;
}

FailureOr<affine::AffineForOp> BroadcastLoopHoist::interchangeLoops(affine::AffineForOp outerLoop,
                                                                    affine::AffineForOp innerLoop) {
  if (!canInterchange(outerLoop, innerLoop)) {
    return failure();
  }

  OpBuilder builder(outerLoop);
  auto loc = outerLoop.getLoc();

  // Create new outer loop with inner loop's bounds (the broadcast dim).
  auto newOuterLoop = builder.create<affine::AffineForOp>(
    loc, innerLoop.getLowerBoundOperands(), innerLoop.getLowerBoundMap(), innerLoop.getUpperBoundOperands(),
    innerLoop.getUpperBoundMap(), innerLoop.getStepAsInt());
  newOuterLoop->setAttrs(innerLoop->getAttrs());

  builder.setInsertionPointToStart(newOuterLoop.getBody());

  // Create new inner loop with outer loop's bounds.
  auto newInnerLoop = builder.create<affine::AffineForOp>(
    loc, outerLoop.getLowerBoundOperands(), outerLoop.getLowerBoundMap(), outerLoop.getUpperBoundOperands(),
    outerLoop.getUpperBoundMap(), outerLoop.getStepAsInt());
  newInnerLoop->setAttrs(outerLoop->getAttrs());

  // Map old IVs to new IVs (swapped).
  IRMapping mapper;
  mapper.map(outerLoop.getInductionVar(), newInnerLoop.getInductionVar());
  mapper.map(innerLoop.getInductionVar(), newOuterLoop.getInductionVar());

  builder.setInsertionPointToStart(newInnerLoop.getBody());

  if (insertIf) {
    auto integerSet = IntegerSet::get(
      /*dimCount=*/2, /*symbolCount=*/0, {builder.getAffineDimExpr(0) - builder.getAffineDimExpr(1)},
      /*eqFlags=*/{false});

    auto ifOp = builder.create<affine::AffineIfOp>(
      loc, integerSet, ValueRange{newInnerLoop.getInductionVar(), newOuterLoop.getInductionVar()},
      /*withElseRegion=*/false);

    builder.setInsertionPointToStart(ifOp.getThenBlock());
  }

  // Clone outer loop's body into new inner loop.
  // When encountering the inner loop, inline its body instead of cloning the loop.
  for (auto &op : outerLoop.getBody()->without_terminator()) {
    if (&op == innerLoop.getOperation()) {
      // Inline the inner loop's body (skip the loop wrapper).
      for (auto &innerOp : innerLoop.getBody()->without_terminator()) {
        builder.clone(innerOp, mapper);
      }
    } else {
      builder.clone(op, mapper);
    }
  }

  // Erase the original outer loop (which contains the original inner loop).
  outerLoop.erase();

  return newOuterLoop;
}

bool BroadcastLoopHoist::hoistBroadcastLoops(func::FuncOp funcOp) {
  bool changed = false;
  bool hoisted = true;

  // Repeatedly find and hoist innermost broadcast loops until no more
  // hoisting is needed. Each iteration may expose new innermost broadcast
  // loops that require hoisting.
  while (hoisted) {
    hoisted = false;

    // Find an innermost broadcast loop that needs hoisting.
    affine::AffineForOp target;
    funcOp.walk([&](affine::AffineForOp forOp) {
      if (!target && isBroadcastLoop(forOp) && isInnermostForOp(forOp) &&
          dyn_cast_or_null<affine::AffineForOp>(forOp->getParentOp()) && getNumLevelsToHoist(forOp) > 0) {
        target = forOp;
      }
    });

    if (!target) {
      break;
    }

    int numLevels = getNumLevelsToHoist(target);

    for (int i = 0; i < numLevels; ++i) {
      auto parentForOp = dyn_cast_or_null<affine::AffineForOp>(target->getParentOp());
      if (!parentForOp) {
        break;
      }
      LLVM_DEBUG(llvm::dbgs() << "Hoisting broadcast loop past parent loop (" << (i + 1) << "/" << numLevels << ")\n");
      auto result = interchangeLoops(parentForOp, target);
      if (failed(result)) {
        break;
      }
      target = *result;
      hoisted = true;
      changed = true;
    }
  }

  return changed;
}

void BroadcastLoopHoist::runOnOperation() {
  auto funcOp = getOperation();

  if (!funcOp->hasAttr(kOperatorTypeStr)) {
    return;
  }

  hoistBroadcastLoops(funcOp);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createBroadcastLoopHoistPass() {
  return std::make_unique<BroadcastLoopHoist>();
}
}  // namespace mlir
