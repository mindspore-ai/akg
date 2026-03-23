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

#include "akg/Dialect/Affine/Transforms/PreProcessForFusion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace mlir {
#define GEN_PASS_DEF_PREPROCESSFORFUSION
#define GEN_PASS_DECL_PREPROCESSFORFUSION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "pre-process-for-fusion"

namespace mlir {
namespace {

struct PreProcessForFusion : public impl::PreProcessForFusionBase<PreProcessForFusion> {
 public:
  PreProcessForFusion() {}

  void runOnOperation() override {
    auto funcOp = getOperation();

    // Move allocations before the first affine.for
    moveAllocBeforeAffineFor(funcOp);

    // Hoist loop invariant loads
    hoistLoopInvariantLoad(funcOp);
  }

 private:
  void moveAllocBeforeAffineFor(func::FuncOp funcOp) {
    Block &block = funcOp.getBody().front();

    Operation *firstAffineFor = nullptr;
    for (Operation &op : block) {
      if (isa<affine::AffineForOp>(&op)) {
        firstAffineFor = &op;
        break;
      }
    }

    if (!firstAffineFor) {
      llvm::errs() << "[AKG] moveAllocBeforeAffineFor: no top-level affine.for found in func " << funcOp.getName()
                   << "\n";
      return;
    }

    SmallVector<Operation *, 8> toMove;

    for (auto it = std::next(firstAffineFor->getIterator()), e = block.end();
         it != e; ++it) {
      Operation *op = &*it;
      if (isa<memref::AllocOp, memref::SubViewOp,
        memref::ReshapeOp, memref::ExpandShapeOp,
        memref::CollapseShapeOp, memref::ReinterpretCastOp,
        memref::MemorySpaceCastOp, arith::IndexCastOp,
        memref::DimOp>(op)) {
        toMove.push_back(op);
      }
    }

    if (toMove.empty()) {
      llvm::errs() << "[AKG] no allocs to move after first top-level affine.for\n";
      return;
    }

    for (Operation *op : toMove) {
      op->moveBefore(firstAffineFor);
    }
  }

  // Helper function to check if a value depends on a loop IV
  bool valueDependsOnIV(Value value, Value iv) {
    if (value == iv) {
      return true;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      return false;
    }

    bool depends = false;
    defOp->walk([&](Operation *op) {
      if (std::any_of(op->getOperands().begin(), op->getOperands().end(),
                     [&](Value operand) { return operand == iv; })) {
        depends = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    return depends;
  }

  // Helper function to check if a memref is invariant to all loops
  bool isMemrefInvariantToAllLoops(affine::AffineLoadOp loadOp,
                                  const SmallVector<affine::AffineForOp> &allEnclosingLoops) {
    Value memref = loadOp.getMemRef();

    for (auto enclosingFor : allEnclosingLoops) {
      Value iv = enclosingFor.getInductionVar();

      // Check if memref depends on loop IV
      if (valueDependsOnIV(memref, iv)) {
        return false;
      }

      // Check if indices depend on loop IV
      for (Value operand : loadOp.getIndices()) {
        if (operand == iv) {
          return false;
        }
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == enclosingFor.getBody()) {
            return false;
          }
        }
      }
    }

    return true;
  }

  // Helper function to check if there are any stores to the memref in the loop subtree
  bool hasStoreToMemrefInLoopTree(affine::AffineForOp forOp, Value memref) {
    bool hasStore = false;
    forOp.walk([&](Operation *op) {
      if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
        if (storeOp.getMemRef() == memref) {
          hasStore = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return hasStore;
  }

  void hoistLoopInvariantLoad(func::FuncOp funcOp) {
    SmallVector<affine::AffineForOp> outermostLoops;
    funcOp.walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>()) {
        outermostLoops.push_back(forOp);
      }
    });

    for (auto outermostLoop : outermostLoops) {
      SmallVector<affine::AffineForOp> nestedLoops;
      outermostLoop.walk([&](affine::AffineForOp forOp) {
        nestedLoops.push_back(forOp);
      });

      SmallVector<Operation *> toHoist;
      for (auto forOp : nestedLoops) {
        for (Operation &op : forOp.getBody()->getOperations()) {
          auto loadOp = dyn_cast<affine::AffineLoadOp>(&op);
          if (!loadOp) continue;

          // Check if the memref is from an alloc operation
          Value memref = loadOp.getMemRef();
          Operation *memrefDefOp = memref.getDefiningOp();
          if (!memrefDefOp || !isa<memref::AllocOp>(memrefDefOp)) {
            continue;
          }

          // Collect all enclosing loops including the current one
          SmallVector<affine::AffineForOp> allEnclosingLoops;
          affine::getAffineForIVs(*loadOp, &allEnclosingLoops);
          allEnclosingLoops.push_back(forOp);

          // Check if memref is invariant to all loops
          if (!isMemrefInvariantToAllLoops(loadOp, allEnclosingLoops)) {
            continue;
          }

          // Check if there are any stores to the same memref in the loop subtree
          if (hasStoreToMemrefInLoopTree(forOp, memref)) {
            continue;
          }

          // Add to hoist list
          toHoist.push_back(loadOp);
        }
      }

      // Move invariant loads to the outermost loop
      for (Operation *op : toHoist) {
        op->moveBefore(outermostLoop);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPreProcessForFusionPass() {
  return std::make_unique<PreProcessForFusion>();
}
}  // namespace mlir
