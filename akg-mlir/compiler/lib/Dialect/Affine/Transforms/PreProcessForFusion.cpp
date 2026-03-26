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

#include <algorithm>

#include "llvm/ADT/DenseMap.h"
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

    // Hoist loop invariant loads and forward scalar store-load pairs
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

    for (auto it = std::next(firstAffineFor->getIterator()), e = block.end(); it != e; ++it) {
      Operation *op = &*it;
      if (isa<memref::AllocOp, memref::SubViewOp, memref::ReshapeOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
              memref::ReinterpretCastOp, memref::MemorySpaceCastOp, arith::IndexCastOp, memref::DimOp>(op)) {
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

  // Collect loop-invariant loads that are candidates for hoisting
  SmallVector<Operation *> collectLoopInvariantLoads(affine::AffineForOp outermostLoop) {
    SmallVector<affine::AffineForOp> nestedLoops;
    outermostLoop.walk([&](affine::AffineForOp forOp) { nestedLoops.push_back(forOp); });

    SmallVector<Operation *> toHoist;
    for (auto forOp : nestedLoops) {
      for (Operation &op : forOp.getBody()->getOperations()) {
        auto loadOp = dyn_cast<affine::AffineLoadOp>(&op);
        if (!loadOp) continue;

        Value memref = loadOp.getMemRef();
        Operation *memrefDefOp = memref.getDefiningOp();
        if (!memrefDefOp || !isa<memref::AllocOp>(memrefDefOp)) {
          continue;
        }

        SmallVector<affine::AffineForOp> allEnclosingLoops;
        affine::getAffineForIVs(*loadOp, &allEnclosingLoops);
        allEnclosingLoops.push_back(forOp);

        if (!isMemrefInvariantToAllLoops(loadOp, allEnclosingLoops)) {
          continue;
        }

        if (hasStoreToMemrefInLoopTree(forOp, memref)) {
          continue;
        }

        toHoist.push_back(loadOp);
      }
    }
    return toHoist;
  }

  // Try to find a scalar store value that can be forwarded to replace a load
  std::pair<Value, affine::AffineStoreOp> tryForwardStoreToLoad(Value memref, affine::AffineForOp outermostLoop) {
    Block *block = outermostLoop->getBlock();
    for (auto it = Block::iterator(outermostLoop); it != block->begin(); --it) {
      auto storeOp = dyn_cast<affine::AffineStoreOp>(&*it);
      if (!storeOp || storeOp.getMemRef() != memref) continue;

      auto memrefType = cast<MemRefType>(memref.getType());
      if (memrefType.getRank() == 0) {
        return {storeOp.getValueToStore(), storeOp};
      }
      break;
    }
    return {Value(), nullptr};
  }

  // Eliminate a load by forwarding a stored value, or fall back to hoisting
  void eliminateOrHoistLoad(Operation *op, affine::AffineForOp outermostLoop) {
    auto loadOp = cast<affine::AffineLoadOp>(op);
    Value memref = loadOp.getMemRef();

    auto [forwardedValue, matchedStore] = tryForwardStoreToLoad(memref, outermostLoop);
    if (!forwardedValue) {
      op->moveBefore(outermostLoop);
      return;
    }

    loadOp.getResult().replaceAllUsesWith(forwardedValue);
    SmallVector<Operation *, 4> toErase;
    toErase.push_back(op);

    if (matchedStore) {
      auto users = memref.getUsers();
      bool memrefStillUsed = std::any_of(
        users.begin(), users.end(), [&](Operation *user) { return user != op && user != matchedStore.getOperation(); });
      if (!memrefStillUsed) {
        toErase.push_back(matchedStore.getOperation());
        Operation *memrefDefOp = memref.getDefiningOp();
        if (memrefDefOp) toErase.push_back(memrefDefOp);
      }
    }

    for (Operation *eraseOp : toErase) {
      eraseOp->erase();
    }
  }

  void hoistLoopInvariantLoad(func::FuncOp funcOp) {
    SmallVector<affine::AffineForOp> outermostLoops;
    funcOp.walk([&](affine::AffineForOp forOp) {
      if (!forOp->getParentOfType<affine::AffineForOp>()) {
        outermostLoops.push_back(forOp);
      }
    });

    for (auto outermostLoop : outermostLoops) {
      SmallVector<Operation *> toHoist = collectLoopInvariantLoads(outermostLoop);
      for (Operation *op : toHoist) {
        eliminateOrHoistLoad(op, outermostLoop);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createPreProcessForFusionPass() {
  return std::make_unique<PreProcessForFusion>();
}
}  // namespace mlir
