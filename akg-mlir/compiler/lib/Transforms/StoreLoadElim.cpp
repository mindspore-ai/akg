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

  bool accessSameLocation(Operation *storeOp, Operation *loadOp) const {
    if (isa<affine::AffineStoreOp>(storeOp) && isa<affine::AffineLoadOp>(loadOp)) {
      affine::MemRefAccess storeAccess(storeOp);
      affine::MemRefAccess loadAccess(loadOp);
      return storeAccess == loadAccess;
    }
    if (isa<memref::StoreOp>(storeOp) && isa<memref::LoadOp>(loadOp)) {
      auto storeIndices = CommonUtils::getStoreLoadIndices(storeOp);
      auto loadIndices = CommonUtils::getStoreLoadIndices(loadOp);
      if (storeIndices.size() != loadIndices.size()) return false;
      for (size_t i = 0; i < storeIndices.size(); ++i) {
        if (storeIndices[i] != loadIndices[i]) return false;
      }
      return true;
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

    for (auto user : memref.getUsers()) {
      if (user == storeOp) {
        continue;
      }
      // Check if user is another store operation, if so, return immediately
      if (isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
        return SmallVector<Operation *>();
      }
      // Check if user is a load operation
      if (isa<memref::LoadOp, affine::AffineLoadOp>(user)) {
        auto storeBlock = storeOp->getBlock();
        auto loadBlock = user->getBlock();
        bool inDiffBranch = (storeBlock != loadBlock);
        bool isNestBranch = inDiffBranch && (storeBlock->getParent() && loadBlock->getParent() &&
                                             storeBlock->getParent()->isAncestor(loadBlock->getParent()));
        bool isSameBranchWAR = !isNestBranch && !inDiffBranch && user->isBeforeInBlock(storeOp);

        // Check if load can be eliminated
        bool canEliminate = !(inDiffBranch && !isNestBranch) && !isSameBranchWAR && accessSameLocation(storeOp, user);
        if (canEliminate) {
          elimLoads.push_back(user);
        }
      }
    }
    return elimLoads;
  }
};

void StoreLoadElimPass::runOnOperation() {
  DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
  SmallVector<Operation *> toElimStores;
  SmallVector<Operation *> toElimLoads;
  // Avoid processing the same load multiple times
  llvm::DenseSet<Operation *> processedLoads;

  getOperation()->walk([&](Operation *op) {
    if (dyn_cast<memref::StoreOp>(op) || dyn_cast<affine::AffineStoreOp>(op)) {
      // check if the memref is valid and the type is correct, skip invalid store operations
      auto memref = CommonUtils::getStoreMemref(op);
      if (!memref || !isa<MemRefType>(memref.getType())) {
        return;
      }
      auto elimLoads = getPossibleElimLoads(op);
      int64_t eraseSize = 0;
      for (auto loadOp : elimLoads) {
        // Skip if load already processed
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
          eraseSize++;
        }
      }
      auto numUsers = std::distance(memref.use_begin(), memref.use_end());
      bool isGlobalBuffer = memref.getDefiningOp() == nullptr;
      bool elimAllLoads = eraseSize > 0 && eraseSize == numUsers - 1;
      if (elimAllLoads && !isGlobalBuffer) {
        toElimStores.push_back(op);
      }
    }
  });

  // Erase load operations first
  for (auto loadOp : toElimLoads) {
    loadOp->erase();
  }

  // Erase store operations and their associated allocations
  for (auto storeOp : toElimStores) {
    // Capture memref before erasing storeOp
    auto memref = CommonUtils::getStoreMemref(storeOp);
    if (storeOp->use_empty()) {
      storeOp->erase();
    }
    // Clean up unused allocations
    if (memref && isa<MemRefType>(memref.getType())) {
      auto memrefOp = memref.getDefiningOp();
      if (memrefOp && memrefOp->use_empty()) {
        memrefOp->erase();
      }
    }
  }
}
}  // end anonymous namespace

std::unique_ptr<Pass> createStoreLoadElimPass() { return std::make_unique<StoreLoadElimPass>(); }
}  // namespace mlir
