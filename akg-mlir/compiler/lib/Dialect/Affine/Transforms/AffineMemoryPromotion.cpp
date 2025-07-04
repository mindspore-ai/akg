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

#include "akg/Dialect/Affine/Transforms/AffineMemoryPromotion.h"

#include <optional>
#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_AFFINEMEMORYPROMOTION
#define GEN_PASS_DECL_AFFINEMEMORYPROMOTION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "affine-memory-promotion"

using namespace mlir;
using namespace mlir::akg;

namespace {
constexpr auto kPrivateCacheLevel = 5;
struct AKGMemoryPromotion : public impl::AffineMemoryPromotionBase<AKGMemoryPromotion> {
  AKGMemoryPromotion() {}
  AKGMemoryPromotion(const std::string &target) : target(target) {}

  void runOnOperation() override;
  void runOnBlock(Block *block, DenseSet<Operation *> &copyNests, unsigned curDepth);

  bool AutoSetPromoteDepth();
  unsigned AutoSetPromoteSpace() const;
  void RemoveGlobalTempBuffer();
  bool DirectlyPromoteGlobalTempBuffer();
  Value MemFlowRemoval(memref::AllocOp memSrc, SmallVector<Operation *> &toErase,
                       SmallVector<Operation *> &toReplace) const;

  DenseSet<Operation *> RemoveEmptyCopyNests(const DenseSet<Operation *> &copyNests) const;

  std::string target = kTargetCuda;
  Value zeroIndex = nullptr;
  bool generateDma{false};
  unsigned slowMemorySpace{0};
  unsigned fastMemorySpace{0};
  unsigned tagMemorySpace{0};
  int minDmaTransferSize{1024};
  int promoteDepth{-1};
  uint64_t fastMemoryCapacity{std::numeric_limits<uint64_t>::max()};
  uint64_t fastMemCapacityBytes{std::numeric_limits<uint64_t>::max()};
};

}  // namespace

/// --------------todo(baiji): START COPY FROM AffineDataCopyGeneration.cpp ---------------------
/// Generate copies for this block. The block is partitioned into separate
/// ranges: each range is either a sequence of one or more operations starting
/// and ending with an affine load or store op, or just an affine.forop (which
/// could have other affine for op's nested within).
void AKGMemoryPromotion::runOnBlock(Block *block, DenseSet<Operation *> &copyNests, unsigned curDepth) {
  if (block->empty()) {
    return;
  }

  uint64_t fastMemCapacityBytes =
    fastMemoryCapacity != std::numeric_limits<uint64_t>::max() ? fastMemoryCapacity * 1024 : fastMemoryCapacity;
  affine::AffineCopyOptions copyOptions = {generateDma, slowMemorySpace, fastMemorySpace, tagMemorySpace,
                                           fastMemCapacityBytes};

  // Every affine.for op in the block starts and ends a block range for copying;
  // in addition, a contiguous sequence of operations starting with a
  // load/store op but not including any copy nests themselves is also
  // identified as a copy block range. Straightline code (a contiguous chunk of
  // operations excluding affine::AffineForOp's) are always assumed to not exhaust
  // memory. As a result, this approach is conservative in some cases at the
  // moment; we do a check later and report an error with location info.

  auto canBePromoted = [&](Operation &op) {
    if (copyNests.count(&op) != 0) {
      return false;
    }
    // An 'affine.if' operation is being treated similar to an
    // operation. 'affine.if''s could have 'affine.for's in them;
    // treat them separately.
    if (auto ifOp = dyn_cast<affine::AffineIfOp>(op)) {
      bool valid = true;
      ifOp->walk([&](Operation *forOp) {
        if (isa<affine::AffineForOp>(forOp)) {
          valid = false;
        }
      });
      return valid;
    }
    return isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineForOp>(op);
  };

  // Get to the first load, store, or for op (that is not a copy nest itself).
  auto curBegin = std::find_if(block->begin(), block->end(), canBePromoted);

  // Create [begin, end) ranges.
  auto it = curBegin;
  while (it != block->end()) {
    affine::AffineForOp forOp;
    // If you hit a non-copy for loop, we will split there.
    if ((forOp = dyn_cast<affine::AffineForOp>(&*it)) && copyNests.count(forOp) == 0) {
      // Perform the copying up unti this 'for' op first.
      (void)affine::affineDataCopyGenerate(curBegin, it, copyOptions, std::nullopt, copyNests);

      // Returns true if the footprint is known to exceed capacity.
      auto exceedsCapacity = [&](affine::AffineForOp forOp) {
        std::optional<int64_t> footprint = getMemoryFootprintBytes(forOp, 0);
        return (footprint.has_value() && static_cast<uint64_t>(*footprint) > fastMemCapacityBytes);
      };

      // If the memory footprint of the 'affine.for' loop is higher than fast
      // memory capacity (when provided), we recurse to copy at an inner level
      // until we find a depth at which footprint fits in fast mem capacity. If
      // the footprint can't be calculated, we assume for now it fits. Recurse
      // inside if footprint for 'forOp' exceeds capacity, or when
      // skipNonUnitStrideLoops is set and the step size is not one.
      bool recurseInner = skipNonUnitStrideLoops ? forOp.getStepAsInt() != 1 : exceedsCapacity(forOp);
      if (promoteDepth != 0 && curDepth < static_cast<unsigned>(promoteDepth)) {
        recurseInner = true;
      }
      if (recurseInner) {
        ++curDepth;
        // We'll recurse and do the copies at an inner level for 'forInst'.
        // Recurse onto the body of this loop.
        runOnBlock(forOp.getBody(), copyNests, curDepth);
      } else {
        // todo(baiji): sometimes there is nullptr in copyNest and don't know why,
        // so remove them before doing affine::affineDataCopyGenerate for now.
        copyNests = RemoveEmptyCopyNests(copyNests);
        // We have enough capacity, i.e., copies will be computed for the
        // portion of the block until 'it', and for 'it', which is 'forOp'. Note
        // that for the latter, the copies are placed just before this loop (for
        // incoming copies) and right after (for outgoing ones).

        // Inner loop copies have their own scope - we don't thus update
        // consumed capacity. The footprint check above guarantees this inner
        // loop's footprint fits.
        (void)affine::affineDataCopyGenerate(it, std::next(it), copyOptions, std::nullopt, copyNests);
      }
      // Get to the next load or store op after 'forOp'.
      curBegin = std::find_if(std::next(it), block->end(), canBePromoted);
      it = curBegin;
    } else {
      assert(copyNests.count(&*it) == 0 && "all copy nests generated should have been skipped above");
      // We simply include this op in the current range and continue for more.
      ++it;
    }
  }

  // Generate the copy for the final block range.
  if (curBegin != block->end()) {
    // Can't be a terminator because it would have been skipped above.
    assert(!curBegin->hasTrait<OpTrait::IsTerminator>() && "can't be a terminator");
    // Exclude the affine.yield - hence, the std::prev.
    (void)affine::affineDataCopyGenerate(curBegin, std::prev(block->end()), copyOptions, std::nullopt, copyNests);
  }
}

void AKGMemoryPromotion::runOnOperation() {
  llvm::outs() << "AKGMemoryPromotion\n";
  if (!AutoSetPromoteDepth()) {
    llvm::outs() << "No need promotion. exit.\n";
    return;
  }
  func::FuncOp f = getOperation();
  OpBuilder topBuilder(f.getBody());
  zeroIndex = topBuilder.create<arith::ConstantIndexOp>(f.getLoc(), 0);

  // Nests that are copy-in's or copy-out's; the root affine::AffineForOps of those
  // nests are stored herein.
  DenseSet<Operation *> copyNests;

  copyNests.clear();

  for (auto &block : f) {
    runOnBlock(&block, copyNests, 0);
  }

  // todo(baiji): sometimes there is nullptr in copyNest and don't know why,
  // so remove them before walking into it for now.
  copyNests = RemoveEmptyCopyNests(copyNests);

  // Promote any single iteration loops in the copy nests and collect
  // load/stores to simplify.
  SmallVector<Operation *, 4> copyOps;
  for (Operation *nest : copyNests) {
    // With a post order walk, the erasure of loops does not affect
    // continuation of the walk or the collection of load/store ops.
    nest->walk([&](Operation *op) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        (void)promoteIfSingleIteration(forOp);
      } else if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
        copyOps.push_back(op);
      }
    });
  }

  // Promoting single iteration loops could lead to simplification of
  // contained load's/store's, and the latter could anyway also be
  // canonicalized.
  RewritePatternSet patterns(&getContext());
  affine::AffineLoadOp::getCanonicalizationPatterns(patterns, &getContext());
  affine::AffineStoreOp::getCanonicalizationPatterns(patterns, &getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  GreedyRewriteConfig config;
  config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
  (void)applyOpPatternsAndFold(copyOps, frozenPatterns, config);

  RemoveGlobalTempBuffer();
}
/// --------------todo(baiji): END COPY FROM AffineDataCopyGeneration.cpp ---------------------

Value AKGMemoryPromotion::MemFlowRemoval(memref::AllocOp memSrc, SmallVector<Operation *> &toErase,
                                         SmallVector<Operation *> &toReplace) const {
  Value memDest;
  for (auto user : memSrc->getUsers()) {
    if (user->use_empty()) {
      if (dyn_cast<affine::AffineStoreOp>(user)) {
        toReplace.push_back(user);
        continue;
      }
    } else {
      // Probably is load gmToFast stmt.
      // Trace the stmt to see if it is copied to fastMem, if so, mark replacement.
      SmallVector<Operation *> gmToFast;
      for (auto store : user->getUsers()) {
        auto memRef = CommonUtils::getStoreMemref(store);
        if (memRef && memRef.getDefiningOp() &&
            CommonUtils::getCacheLevel(cast<TypedValue<MemRefType>>(memRef)) != kGlobalCache) {
          memDest = memRef;
          gmToFast.push_back(store);
        }
      }
      for (auto stmt : gmToFast) {
        stmt->erase();
      }
    }

    // After all earse, check use_empty again.
    if (user->use_empty()) {
      toErase.push_back(user);
    }
  }
  return memDest;
}

void AKGMemoryPromotion::RemoveGlobalTempBuffer() {
  auto tempBuffers = CommonUtils::findTempBuffer(getOperation());
  auto it = tempBuffers.find(kGlobalCache);
  if (it == tempBuffers.end()) {
    return;
  }
  for (auto tempGm : it->second) {
    SmallVector<Operation *> toErase;
    SmallVector<Operation *> toReplace;
    Value tempFastMem = MemFlowRemoval(tempGm, toErase, toReplace);
    if (!tempFastMem || !tempFastMem.getDefiningOp()) {
      continue;
    }
    for (auto user : toErase) {
      user->erase();
    }
    for (auto user : toReplace) {
      bool isFastMemToGlobal = false;
      getOperation()->walk([&](affine::AffineLoadOp load) {
        for (auto loadUser : load->getUsers()) {
          isFastMemToGlobal = isFastMemToGlobal || (loadUser == user && tempFastMem == load.getMemref());
        }
      });

      if (isFastMemToGlobal) {
        // todo(baiji): can futher remove the load and dealloc of fastMem as well.
        user->erase();
        continue;
      }

      auto memRef = CommonUtils::getStoreMemref(user);
      if (memRef && memRef.getDefiningOp()) {
        memRef.replaceAllUsesWith(tempFastMem);
      }
    }
    if (tempGm->use_empty()) {
      tempGm->erase();
    }
  }
}

DenseSet<Operation *> AKGMemoryPromotion::RemoveEmptyCopyNests(const DenseSet<Operation *> &copyNests) const {
  DenseSet<Operation *> cleanCopyNests;
  for (auto cp : copyNests) {
    if (cp == nullptr) {
      continue;
    }
    (void)cleanCopyNests.insert(cp);
  }
  return cleanCopyNests;
}

unsigned AKGMemoryPromotion::AutoSetPromoteSpace() const { return kPrivateCacheLevel; }

bool AKGMemoryPromotion::DirectlyPromoteGlobalTempBuffer() {
  auto funcOp = getOperation();
  auto redOp = CommonUtils::getReduceOps(funcOp)[0];
  auto redDirection = CommonUtils::getReduceDirection(redOp);
  if (redDirection != ReduceDirection::ALL) {
    return false;
  }
  auto redAlloc = CommonUtils::getAllocOpOfValue(funcOp, redOp->getOperands()[1]);
  if (redAlloc == nullptr) {
    return false;
  }
  auto tempBuffers = CommonUtils::findTempBuffer(funcOp);
  auto it = tempBuffers.find(kGlobalCache);
  if (it == tempBuffers.end()) {
    return false;
  }
  for (auto tempGm : it->second) {
    if (tempGm != redAlloc) {
      continue;
    }
    auto origType = tempGm.getType().dyn_cast<MemRefType>();
    auto newType = MemRefType::get(origType.getShape(), origType.getElementType(), {}, fastMemorySpace);
    for (auto *user : redAlloc->getUsers()) {
      for (auto operand : user->getOperands()) {
        if (operand.getType() == origType) {
          operand.setType(newType);
        }
      }
    }
    return true;
  }
  return false;
}

bool AKGMemoryPromotion::AutoSetPromoteDepth() {
  int lastRedAxisDepth = -1;
  int currDepth = 0;
  // walk from inner to outer ?
  getOperation()->walk([&](affine::AffineForOp forOp) {
    if (lastRedAxisDepth == -1 && CommonUtils::isReduceAxis(getOperation(), forOp)) {
      lastRedAxisDepth = currDepth;
    }
    currDepth++;
  });
  if (lastRedAxisDepth != -1) {
    // promote below the innermost reduction axis to local memory
    promoteDepth = currDepth - lastRedAxisDepth;
  }
  llvm::outs() << "LastRedAxisDepth = " << lastRedAxisDepth << " total " << currDepth << " promoteDepth "
               << promoteDepth << "\n";
  if (promoteDepth < 0) {
    return false;
  }
  fastMemorySpace = AutoSetPromoteSpace();

  // For Reduce-All case, we can directly promote global-temp buffer to our dest mem space.
  if (DirectlyPromoteGlobalTempBuffer()) {
    return false;
  }
  // todo(baiji): check forOp's size, if 1, return false;
  return true;
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineMemoryPromotionPass(const std::string &target) {
  return std::make_unique<AKGMemoryPromotion>(target);
}
std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAffineMemoryPromotionPass() {
  return std::make_unique<AKGMemoryPromotion>();
}
