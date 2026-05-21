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

#include "akg/Transforms/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DECL_STORELOADELIM
#define GEN_PASS_DECL_STORELOADELIM
#ifndef GEN_PASS_DEF_STORELOADELIM
#define GEN_PASS_DEF_STORELOADELIM
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "elim-store-load"

namespace mlir {
namespace {

// ===----------------------------------------------------------------------===//
// StoreLoadElimPass
// ===----------------------------------------------------------------------===//

// This pass removes redundant load store pairs for temp buffers.
// e.g. the affine.load/store can be removed because we can replace all uses of %4 by %3
// %3 = arith.mulf %2, %2 : f32
// affine.store %3, %alloc[%arg3, %arg4] : memref<128x768xf32>
// %4 = affine.load %alloc[%arg3, %arg4] : memref<128x768xf32>

// And we cannot remove them when:
// 1. the store is not only used by load (e.g. will be used by mem copy if store dest is the global output)
//  e.g.
//    store %x, alloc
//    load alloc
//    ...
//    memref.copy alloc, %global_out
// 2. store and load are not located in the same branch (whatever comes from different If or For)
//    because the stored variable %x will be out of scope for load
//  e.g.
//    for arg0
//      store %x alloc
//    load alloc[]
// 3. load are ahead of store (WAR cases)
//  e.g.
//    %3 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>
//    %4 = arith.addf %2, %3 {reduction_axes = [2 : index], reduction_type = "x"} : f32
//    affine.store %4, %alloc[%arg2, %arg3] : memref<8x128xf32>
//    %0 = affine.load %alloc[%arg2, %arg3] : memref<8x128xf32>

struct StoreLoadElimPass : public StoreLoadElimBase<StoreLoadElimPass> {
 public:
  void runOnOperation() override;

 private:
  Value getLoadResult(Operation *loadOp) const {
    Value loadResult;
    if (dyn_cast<affine::AffineLoadOp>(loadOp)) {
      loadResult = dyn_cast<affine::AffineLoadOp>(loadOp).getResult();
    } else if (dyn_cast<memref::LoadOp>(loadOp)) {
      loadResult = dyn_cast<memref::LoadOp>(loadOp).getResult();
    } else {
      assert(false && "can only get result from AffineLoad or memref::LoadOp.");
    }
    return loadResult;
  }

  bool accessSameLocation(Operation *op1, Operation *op2) const {
    SmallVector<Value, 4> vals1, vals2;
    if (isa<affine::AffineStoreOp, affine::AffineLoadOp>(op1) &&
        isa<affine::AffineStoreOp, affine::AffineLoadOp>(op2)) {
      AffineMap map1, map2;
      CommonUtils::getUnifiedAffineAccess(op1, map1, vals1);
      CommonUtils::getUnifiedAffineAccess(op2, map2, vals2);
      if (!map1 || !map2 || map1 != map2) return false;
    } else if (isa<memref::StoreOp, memref::LoadOp>(op1) && isa<memref::StoreOp, memref::LoadOp>(op2)) {
      llvm::append_range(vals1, CommonUtils::getStoreLoadIndices(op1));
      llvm::append_range(vals2, CommonUtils::getStoreLoadIndices(op2));
    } else {
      return false;
    }
    if (vals1.size() != vals2.size()) return false;
    for (size_t i = 0; i < vals1.size(); ++i) {
      if (vals1[i] != vals2[i]) return false;
    }
    return true;
  }

  // Returns true if any store in otherStores writes to the same location as loadOp
  // and is executed strictly between storeOp and loadOp (would shadow the forward).
  // Conservative across blocks: if the intervening store is in a different block,
  // we treat it as intervening.
  bool hasInterveningStore(Operation *storeOp, Operation *loadOp, const SmallVector<Operation *> &otherStores) const {
    auto storeBlock = storeOp->getBlock();
    auto loadBlock = loadOp->getBlock();
    for (auto otherStore : otherStores) {
      if (!accessSameLocation(otherStore, loadOp)) continue;
      auto otherBlock = otherStore->getBlock();
      if (otherBlock == storeBlock && loadBlock == storeBlock) {
        if (storeOp->isBeforeInBlock(otherStore) && otherStore->isBeforeInBlock(loadOp)) {
          return true;
        }
      } else {
        return true;
      }
    }
    return false;
  }

  SmallVector<Operation *> getPossibleElimLoads(Operation *storeOp) const {
    SmallVector<Operation *> elimLoads;
    auto memref = CommonUtils::getStoreMemref(storeOp);
    // check if the memref is valid and the type is correct
    if (!memref || !isa<MemRefType>(memref.getType())) {
      return SmallVector<Operation *>();
    }

    // Partition users into other stores vs. candidate loads.
    // Other users (memref.copy, etc.) don't block forwarding but may block alloc cleanup.
    SmallVector<Operation *> otherStores;
    SmallVector<Operation *> candidateLoads;
    for (auto user : memref.getUsers()) {
      if (user == storeOp) {
        continue;
      }
      if (isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
        otherStores.push_back(user);
      } else if (isa<memref::LoadOp, affine::AffineLoadOp>(user)) {
        candidateLoads.push_back(user);
      }
    }

    for (auto loadOp : candidateLoads) {
      auto storeBlock = storeOp->getBlock();
      auto loadBlock = loadOp->getBlock();
      bool inDiffBranch = (storeBlock != loadBlock);
      bool isNestBranch = inDiffBranch && (storeBlock->getParent() && loadBlock->getParent() &&
                                           storeBlock->getParent()->isAncestor(loadBlock->getParent()));
      bool isSameBranchWAR = !isNestBranch && !inDiffBranch && loadOp->isBeforeInBlock(storeOp);

      bool canEliminate = !(inDiffBranch && !isNestBranch) && !isSameBranchWAR && accessSameLocation(storeOp, loadOp);
      if (!canEliminate) continue;
      if (hasInterveningStore(storeOp, loadOp, otherStores)) continue;
      elimLoads.push_back(loadOp);
    }
    return elimLoads;
  }
};

void StoreLoadElimPass::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  SmallVector<Operation *> toElimLoads;
  // Avoid processing the same load multiple times
  llvm::DenseSet<Operation *> processedLoads;
  // Memrefs that had at least one load forwarded — candidates for store/alloc cleanup.
  llvm::SmallDenseSet<Value> affectedMemrefs;

  getOperation()->walk([&](Operation *op) {
    if (!isa<memref::StoreOp, affine::AffineStoreOp>(op)) {
      return;
    }
    auto memref = CommonUtils::getStoreMemref(op);
    if (!memref || !isa<MemRefType>(memref.getType())) {
      return;
    }

    auto elimLoads = getPossibleElimLoads(op);
    for (auto loadOp : elimLoads) {
      if (processedLoads.count(loadOp)) {
        continue;
      }
      processedLoads.insert(loadOp);

      Value storeValue = CommonUtils::getStoreValue(op);
      if (!domInfo.properlyDominates(storeValue, loadOp)) {
        continue;
      }
      Value loadResult = getLoadResult(loadOp);
      loadResult.replaceAllUsesWith(storeValue);
      if (loadOp->use_empty()) {
        toElimLoads.push_back(loadOp);
        affectedMemrefs.insert(memref);
      }
    }
  });

  // Erase load operations first
  for (auto loadOp : toElimLoads) {
    loadOp->erase();
  }

  // If all loads were forwarded and remaining users are only stores, the buffer
  // is dead — erase those stores and the allocation.
  for (auto memref : affectedMemrefs) {
    auto memrefOp = memref.getDefiningOp();
    if (!memrefOp) continue;  // function arg / global buffer
    SmallVector<Operation *> storesToErase;
    bool onlyStores = true;
    for (auto user : memref.getUsers()) {
      if (isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
        storesToErase.push_back(user);
      } else {
        onlyStores = false;
        break;
      }
    }
    if (!onlyStores) continue;
    for (auto storeOp : storesToErase) {
      storeOp->erase();
    }
    if (memrefOp->use_empty()) {
      memrefOp->erase();
    }
  }
}
}  // end anonymous namespace

std::unique_ptr<Pass> createStoreLoadElimPass() { return std::make_unique<StoreLoadElimPass>(); }
}  // namespace mlir
