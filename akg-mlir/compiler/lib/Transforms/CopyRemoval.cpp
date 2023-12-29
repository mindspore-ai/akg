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

#include "akg/Transforms/CopyRemoval.h"

#include <algorithm>
#include <iterator>
#include "akg/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DECL_COPYREMOVAL
#define GEN_PASS_DECL_COPYREMOVAL
#ifndef GEN_PASS_DEF_COPYREMOVAL
#define GEN_PASS_DEF_COPYREMOVAL
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "copy-removal"

using namespace mlir;
using namespace MemoryEffects;

namespace {

// ===----------------------------------------------------------------------===//
// CopyRemovalPass
// ===----------------------------------------------------------------------===//

/// This pass removes redundant copy operations. Additionally, it
/// removes leftover definition and deallocation operations by erasing the
/// copy operation.
/// func() {
///   %source = alloc()
///   write_to(%source)
///   %destination = alloc()
///   copy(%source, %destination)
///   dealloc(%source)
///   return %destination
/// }
///
/// Output:
/// func(){
///   %source = alloc()
///   write_to(%source)
///   return %source
/// }
/// Constraints:
/// 1) There should not exist any users of `destination` before the copy op.
/// 2) There should not be any write operations on `source` and `destination`
/// after copy op.

// todo: Modifying the common code of the community
struct CopyRemovalPass : public CopyRemovalBase<CopyRemovalPass> {
 public:
  void runOnOperation() override;

 private:
  /// Returns the allocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getAllocationOp(Value value) const {
    if (Operation *op = value.getDefiningOp()) {
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (effects.hasEffect<Allocate>()) {
          return op;
        }
      }
    }
    return nullptr;
  }

  /// Returns the deallocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getDeallocationOp(Value value) const {
    auto valueUsers = value.getUsers();
    auto it = llvm::find_if(valueUsers, [&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      return effects && effects.hasEffect<Free>();
    });
    return (it == valueUsers.end() ? nullptr : *it);
  }

  /// Check whether the write effect on `val` can be caused by `op`.
  static bool doesOpHaveWriteEffect(Value val, Operation *op) {
    // Check whether the operation `op` has write effect on the memory.
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!llvm::is_contained(val.getUsers(), op)) {
        return false;
      }
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      // Iterate through all the effects produced by `op`, and check
      // whether one of them is MemoryEffects::Write.
      return llvm::any_of(effects, [](const MemoryEffects::EffectInstance effect) {
        return isa<MemoryEffects::Write>(effect.getEffect());
      });
    }

    if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
      // Recurse into the regions for this op and check whether the
      // internal operations may have the side effect `EffectType` on
      // `val`.
      for (Region &region : op->getRegions()) {
        auto walkResult = region.walk([&](Operation *op) {
          if (doesOpHaveWriteEffect(val, op)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (walkResult.wasInterrupted()) {
          return true;
        }
      }
      return false;
    }

    // Otherwise, conservatively assume generic operations have the effect
    // on the operation.
    return true;
  }

  /// Check whether the write effect on `val` can be caused by `op`.
  static bool doesOpUseVal(Value val, Operation *op) {
    if (!llvm::is_contained(val.getUsers(), op)) {
      return false;
    }
    return true;
  }

  /// Check if an op that lies on one of the paths between `start`
  /// and `end` and satisfies `checkPropertiesOfOperation`. If the start and end
  /// operations are in different regions, recursively consider all path from
  /// `start` to the parent of `end` and all paths from the parent of `end` to
  /// `end`. When `start` and `end` exist in the same region, perform a CFG
  /// traversal to check all the relevant operations.
  bool hasInterveningOp(const Value val, Operation *start, Operation *end,
                        std::function<bool(Value, Operation *)> checkPropertiesOfOperation) const {
    // Check for all paths from operation `fromp` to operation `untilOp` for the
    // given property.
    std::function<bool(Operation *, Operation *)> recur = [&](Operation *fromOp, Operation *untilOp) {
      auto untilOpParentRegion = untilOp->getParentRegion();
      auto untilOpParentOp = untilOp->getParentOp();
      auto fromOpParentRegion = fromOp->getParentRegion();
      auto fromOpBlock = fromOp->getBlock();
      auto untilOpBlock = untilOp->getBlock();

      if (!fromOpParentRegion->isAncestor(untilOpParentRegion)) {
        return false;
      }

      // If the operations are in different regions, recursively consider
      // all path from `fromOp` to the parent of `untilOp` and all paths
      // from the parent of `untilOp` to `untilOp`.
      if (fromOpParentRegion != untilOpParentRegion) {
        (void)recur(fromOp, untilOpParentOp);
        if (checkPropertiesOfOperation(val, untilOpParentOp)) {
          return true;
        }
        return false;
      }

      // Now, assuming that `fromOp` and `untilOp` exist in the same region,
      // perform a CFG traversal to check all the relevant operations.

      // Additional blocks to consider.
      SmallVector<Block *, 2> todoBlocks;
      {
        // First consider the parent block of `fromOp` an check all
        // operations after `from`.
        for (auto iter = ++fromOp->getIterator(), end = fromOpBlock->end(); iter != end && &*iter != untilOp; ++iter) {
          if (checkPropertiesOfOperation(val, &*iter)) {
            return true;
          }
        }

        // If the parent of `fromOp` doesn't contain `untilOp`, add the
        // successors to the list of blocks to check.
        if (untilOpBlock != fromOpBlock) {
          (void)std::copy(fromOpBlock->getSuccessors().begin(), fromOpBlock->getSuccessors().end(),
                          std::back_inserter(todoBlocks));
        }
      }

      // Stores the blocks whose ops has been checked using
      // `checkPropertiesOfOperation`.
      SmallPtrSet<Block *, 4> done;
      // Traverse the CFG until hitting `untilOp`.
      while (!todoBlocks.empty()) {
        Block *blk = todoBlocks.pop_back_val();
        if (done.insert(blk).second) {
          continue;
        }
        for (Operation &op : *blk) {
          if (&op == untilOp) {
            break;
          }
          if (checkPropertiesOfOperation(val, &op)) {
            return true;
          }
          if (&op == blk->getTerminator()) {
            (void)std::copy(blk->getSuccessors().begin(), blk->getSuccessors().end(), std::back_inserter(todoBlocks));
          }
        }
      }
      return false;
    };
    return recur(start, end);
  }

  void replaceDest4StoreOp(CopyOpInterface copyOp) const {
    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();
    src.replaceAllUsesWith(dest);
  }

  /// Remove copy statements when there are no uses of `destination` after the
  /// copy op.
  void removeCopy(CopyOpInterface copyOp, llvm::SmallPtrSet<Operation *, 4> &opsToErase) {
    if (opsToErase.count(copyOp) != 0) {
      return;
    }
    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();
    if (src.getDefiningOp() == nullptr && dest.getDefiningOp() == nullptr ||
        isa<memref::GetGlobalOp>(src.getDefiningOp()) || isa<memref::ExpandShapeOp>(src.getDefiningOp()) ||
        isa<memref::CollapseShapeOp>(src.getDefiningOp()) || isa<memref::ReshapeOp>(src.getDefiningOp()) ||
        isa<memref::SubViewOp>(src.getDefiningOp())) {
      return;
    }
    Operation *lastOpUsingDest = &src.getParentRegion()->back().back();
    Operation *srcDeallocOp = getDeallocationOp(src);
    Operation *destDeallocOp = getDeallocationOp(dest);
    if (srcDeallocOp) {
      (void)opsToErase.insert(srcDeallocOp);
    }
    if (destDeallocOp) {
      lastOpUsingDest = destDeallocOp;
    }
    if (!hasInterveningOp(dest, copyOp, lastOpUsingDest, &doesOpUseVal) &&
        (!doesOpUseVal(dest, lastOpUsingDest) || destDeallocOp)) {
      (void)opsToErase.insert(copyOp);
    }
    (void)opsToErase.insert(src.getDefiningOp());
    replaceDest4StoreOp(copyOp);
  }
};

void CopyRemovalPass::runOnOperation() {
  /// Operations that need to be removed.
  SmallVector<func::FuncOp, 2> funcs;
  getOperation()->walk([&](func::FuncOp func) { funcs.push_back(func); });
  for (auto func : funcs) {
    llvm::SmallPtrSet<Operation *, 4> opsToErase;
    func.walk([&](CopyOpInterface copyOp) { removeCopy(copyOp, opsToErase); });
    for (Operation *op : opsToErase) {
      assert(op->use_empty() &&
             "uses remaining for copy ops, memref allocation and deallocation "
             "ops that should have ready to be erased");
      op->erase();
    }
  }
  return;
}

}  // end anonymous namespace

std::unique_ptr<Pass> mlir::createCopyRemovalPass() { return std::make_unique<CopyRemovalPass>(); }
