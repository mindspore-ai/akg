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

// ===- CopyElision.cc - Removes unnecessary copies -------------------------=== //
//
//
// ===----------------------------------------------------------------------=== //

#include "akg/Transforms/CopyElision.h"
#include <algorithm>

#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"


namespace mlir {
#ifndef GEN_PASS_DEF_COPYELISION
#define GEN_PASS_DEF_COPYELISION
#include "akg/Transforms/Passes.h.inc"
#endif
}  // namespace mlir

#define DEBUG_TYPE "copy-elision"

namespace mlir {

static constexpr const int kVectorSizeFour = 4;

// ===----------------------------------------------------------------------=== //
// CopyElisionPass
// ===----------------------------------------------------------------------=== //

/// This pass removes unnecessary copy operations. Additionally, it
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
/// 1) There should not exist any read of `destination` before the copy op.
/// 2) If there is some write operations on `destination`, there should not exist
/// any read or write of `source`
/// 3) If there is some read operations on `destination`, there should not exist
/// any write of the `source`
struct CopyElisionPass : public mlir::impl::CopyElisionBase<CopyElisionPass> {
 public:
  void runOnOperation() override;

  CopyElisionPass() = default;
  CopyElisionPass(const CopyElisionPass &pass) = default;

 private:
  /// Returns the allocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getAllocationOp(const Value &value) const {
    if (Operation *op = value.getDefiningOp()) {
      if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
        if (effects.hasEffect<MemoryEffects::Allocate>()) {
          return op;
        }
      }
    }
    return nullptr;
  }

  /// Returns the deallocation operation for `value` if it exists.
  /// nullptr otherwise.
  Operation *getDeallocationOp(const Value &value) const {
    auto valueUsers = value.getUsers();
    auto it = llvm::find_if(valueUsers, [&](Operation *op) {
      auto effects = dyn_cast<MemoryEffectOpInterface>(op);
      return effects && effects.hasEffect<MemoryEffects::Free>();
    });
    return (it == valueUsers.end() ? nullptr : *it);
  }

  /// Check whether the write effect on `val` can be caused by `op`.
  static bool doesOpHaveWriteEffect(const Value &val, Operation *op) {
    // Check whether the operation `op` has write effect on the memory.
    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      if (!llvm::is_contained(val.getUsers(), op)) {
        return false;
      }
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      // Iterate through all the effects produced by `op`, and check
      // whether one of them is MemoryEffects::Write.
      return llvm::any_of(effects, [](const MemoryEffects::EffectInstance &effect) {
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
  static bool doesOpUseVal(const Value &val, Operation *op) {
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
  bool hasInterveningOp(const Value &val, Operation *start, Operation *end,
                        const std::function<bool(Value, Operation *)> &checkPropertiesOfOperation) const {
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
          auto succ = fromOpBlock->getSuccessors();
          std::copy(succ.begin(), succ.end(), std::back_inserter(todoBlocks));
        }
      }

      // Stores the blocks whose ops has been checked using
      // `checkPropertiesOfOperation`.
      SmallPtrSet<Block *, kVectorSizeFour> done;
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
            auto succ = blk->getSuccessors();
            std::copy(succ.begin(), succ.end(), std::back_inserter(todoBlocks));
          }
        }
      }
      return false;
    };
    return recur(start, end);
  }

  /// Replace all occurrences of `destination` with `source` if the following
  /// conditions are met.
  /// 1) There should not exist any users of `destination` before the copy op.
  /// 2) If there is some write operations on `destination`, there should not exist
  /// any uses or write of `source`
  /// 3) If there is some read operations on `destination`, there should not exist
  /// any write of the `source`

  bool canReplaceDestWithSrc(
      CopyOpInterface copyOp,
      Value src,
      Value dest,
      Operation* srcDeallocOp,
      Operation* destDefOp,
      Operation* lastOpOfCurrentRegion) const {
      Operation *lastOpUsingSrc = lastOpOfCurrentRegion;

      // If `srcDeallocOp` is not null, `lastOpUsingSrc` will be `srcDeallocOp`.
      if (srcDeallocOp) {
        lastOpUsingSrc = srcDeallocOp;
      }
      Operation *firstOpUsingDest = &dest.getParentRegion()->front().front();

      // If `destDefOp` is not null, `firstOpUsingDest` will be `destDefOp`.
      if (destDefOp) {
        firstOpUsingDest = destDefOp;
      }

      bool isDestReadBefore =
        hasInterveningOp(dest, firstOpUsingDest, copyOp, &doesOpUseVal) || doesOpUseVal(dest, firstOpUsingDest);
      bool isDestWriteAfter = hasInterveningOp(dest, copyOp, lastOpUsingSrc, &doesOpHaveWriteEffect) ||
                              doesOpHaveWriteEffect(dest, lastOpUsingSrc);
      bool isSrcWriteAfter = hasInterveningOp(src, copyOp, lastOpUsingSrc, &doesOpHaveWriteEffect) ||
                            doesOpHaveWriteEffect(src, lastOpUsingSrc);
      bool isSrcReadAfter =
        hasInterveningOp(src, copyOp, lastOpUsingSrc, &doesOpUseVal) || doesOpUseVal(src, lastOpUsingSrc);
      bool isDestReadAfter =
        hasInterveningOp(dest, copyOp, lastOpUsingSrc, &doesOpUseVal) || doesOpUseVal(dest, lastOpUsingSrc);
      // Capture all the cases when copy removal and replacing uses of `dest` with
      // `src` is not possible
      // 1. Check if dest value is used before the copy
      // 2. Check if (src value is write or read after the copy) and dest value is write after the copy
      // 3. Check if (src value is write after the copy) and dest value is read after the copy
      // if (isDestReadBefore ||
      //     ((isSrcWriteAfter || isSrcReadAfter) && isDestWriteAfter) ||
      //     (isSrcWriteAfter && isDestReadAfter)
      //     )
      if (isDestReadBefore || (isSrcWriteAfter && (isDestReadAfter || isDestWriteAfter)) ||
          (isSrcReadAfter && isDestWriteAfter)) {
        return false;
      }
      return true;
    }

  bool isMemRefTypesCompatible(Value src, Value dest) const {
    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto destType = dyn_cast<MemRefType>(dest.getType());
    if (!destType || !srcType) {
      return true;
    }

    if (srcType.getShape() != destType.getShape() ||
        srcType.getElementType() != destType.getElementType() ||
        srcType.getMemorySpace() != destType.getMemorySpace()) {
      return false;
    }

    auto srcLayout = srcType.getLayout();
    auto destLayout = destType.getLayout();

    if (!srcLayout && !destLayout) {
      return true;
    }
    if ((!srcLayout && destLayout) || (srcLayout && !destLayout)) {
      return false;
    }

    return srcLayout == destLayout;
  }

  void reuseCopySourceAsTarget(CopyOpInterface copyOp,
                               llvm::SmallPtrSet<Operation *, kVectorSizeFour> &opsToErase) const {
    if (opsToErase.count(copyOp) != 0) {
      return;
    }

    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();

    Operation *srcDeallocOp = getDeallocationOp(src);
    Operation *destDeallocOp = getDeallocationOp(dest);
    Operation *destDefOp = getAllocationOp(dest);
    Operation *lastOpOfCurrentRegion = &src.getParentRegion()->back().back();

    // Check if a replacement of `dest` with `src` is possible.
    if (!canReplaceDestWithSrc(copyOp, src, dest, srcDeallocOp, destDefOp, lastOpOfCurrentRegion)) {
      return;
    }
    // Check if a cast is needed.
    Value replacement = src;

    if (!isMemRefTypesCompatible(src, dest)) {
      OpBuilder builder(copyOp);
      builder.setInsertionPointAfter(copyOp);
      replacement = builder.create<memref::CastOp>(copyOp.getLoc(), dest.getType(), src);
    }

    // Erase the `copyOp`, `destDefOp` and `destDeallocOp`. Also remove
    // `srcDeallocOp` if any uses of `dest` are there after `srcDeallocOp`, as
    // we are replacing all instances of `dest` with `src`, and doing so will
    // lead to occurrences of `src` after `srcDeallocOp`, which is semantically
    // incorrect.
    (void)opsToErase.insert(copyOp);
    if (destDefOp) {
      (void)opsToErase.insert(destDefOp);
    }
    if (srcDeallocOp && (hasInterveningOp(dest, srcDeallocOp, lastOpOfCurrentRegion, &doesOpUseVal) ||
                         doesOpUseVal(dest, lastOpOfCurrentRegion))) {
      (void)opsToErase.insert(srcDeallocOp);
    }
    if (destDeallocOp) {
      (void)opsToErase.insert(destDeallocOp);
    }

    // Replace all uses of `src` or cast with `dest`.
    dest.replaceAllUsesWith(replacement);
  }

  /// Remove copy statements when there are no uses of `destination` after the
  /// copy op.
  void removeCopy(CopyOpInterface copyOp, llvm::SmallPtrSet<Operation *, kVectorSizeFour> &opsToErase) const {
    if (opsToErase.count(copyOp) != 0) {
      return;
    }

    Value src = copyOp.getSource();
    Value dest = copyOp.getTarget();
    Operation *lastOpUsingDest = &src.getParentRegion()->back().back();
    Operation *destDeallocOp = getDeallocationOp(dest);
    if (destDeallocOp) {
      lastOpUsingDest = destDeallocOp;
    }
    if (!hasInterveningOp(dest, copyOp, lastOpUsingDest, &doesOpUseVal) &&
        (!doesOpUseVal(dest, lastOpUsingDest) || destDeallocOp)) {
      (void)opsToErase.insert(copyOp);
    }
  }
};

void CopyElisionPass::runOnOperation() {
  /// Operations that need to be removed.
  llvm::SmallPtrSet<Operation *, kVectorSizeFour> opsToErase;
  getOperation()->walk([&](CopyOpInterface copyOp) {
    if (isa<BlockArgument>(copyOp.getTarget())) {
      return;
    }
    reuseCopySourceAsTarget(copyOp, opsToErase);
    removeCopy(copyOp, opsToErase);
  });
  for (Operation *op : opsToErase) {
    assert(op->use_empty() &&
           "uses remaining for copy ops, memref allocation and deallocation "
           "ops that should have ready to be erased");
    op->erase();
  }
  return;
}

}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::createCopyElisionPass() { return std::make_unique<mlir::CopyElisionPass>(); }
