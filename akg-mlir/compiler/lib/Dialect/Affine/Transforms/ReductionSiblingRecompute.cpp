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

#include "akg/Dialect/Affine/Transforms/ReductionSiblingRecompute.h"

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
#ifndef GEN_PASS_DEF_REDUCTIONSIBLINGRECOMPUTE
#define GEN_PASS_DEF_REDUCTIONSIBLINGRECOMPUTE
#ifndef GEN_PASS_DECL_REDUCTIONSIBLINGRECOMPUTE
#define GEN_PASS_DECL_REDUCTIONSIBLINGRECOMPUTE
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "reduction-sibling-recompute"

namespace mlir {
namespace {

struct ReductionSiblingRecomputePass : public impl::ReductionSiblingRecomputeBase<ReductionSiblingRecomputePass> {
 public:
  void runOnOperation() override;

 private:
  using ValueMap = llvm::DenseMap<Value, Value>;
  struct FuncScanInfo {
    SmallVector<affine::AffineForOp> outerLoops;
    SmallVector<memref::AllocOp> allocs;
    llvm::DenseSet<Value> allocResults;
  };
  struct RecomputeCandidate {
    affine::AffineForOp srcLoop;
    affine::AffineForOp dstLoop;
    affine::AffineLoadOp dstLoad;
    affine::AffineStoreOp srcStore;
  };

  static bool sameAccessInBlock(Operation *storeOp, Operation *loadOp) {
    if (CommonUtils::getStoreMemref(storeOp) != loadOp->getOperand(0)) {
      return false;
    }
    auto storeIndices = CommonUtils::getStoreLoadIndices(storeOp);
    auto loadIndices = CommonUtils::getStoreLoadIndices(loadOp);
    if (storeIndices.size() != loadIndices.size()) {
      return false;
    }
    for (auto [storeIdx, loadIdx] : llvm::zip(storeIndices, loadIndices)) {
      if (storeIdx != loadIdx) {
        return false;
      }
    }
    auto affineStore = dyn_cast<affine::AffineStoreOp>(storeOp);
    auto affineLoad = dyn_cast<affine::AffineLoadOp>(loadOp);
    if (affineStore || affineLoad) {
      return affineStore && affineLoad && affineStore.getAffineMap() == affineLoad.getAffineMap();
    }
    return isa<memref::StoreOp>(storeOp) && isa<memref::LoadOp>(loadOp);
  }

  static bool sameSiblingAccess(affine::AffineStoreOp storeOp, affine::AffineForOp srcLoop, affine::AffineLoadOp loadOp,
                                affine::AffineForOp dstLoop) {
    if (storeOp.getMemRef() != loadOp.getMemRef() || storeOp.getAffineMap() != loadOp.getAffineMap()) {
      return false;
    }
    auto storeIndices = storeOp.getIndices();
    auto loadIndices = loadOp.getIndices();
    if (storeIndices.size() != loadIndices.size()) {
      return false;
    }
    for (auto [storeIdx, loadIdx] : llvm::zip(storeIndices, loadIndices)) {
      if (storeIdx == loadIdx) {
        continue;
      }
      if (storeIdx == srcLoop.getInductionVar() && loadIdx == dstLoop.getInductionVar()) {
        continue;
      }
      return false;
    }
    return true;
  }

  static bool isSupportedCloneOp(Operation *op) {
    if (!op || op->getNumRegions() != 0 || op->getNumResults() == 0) {
      return false;
    }
    if (isa<affine::AffineLoadOp, memref::LoadOp>(op)) {
      return true;
    }
    return isMemoryEffectFree(op);
  }
  static bool isCloneTransparentOp(Operation *op) {
    if (!op || op->getNumRegions() != 0) {
      return false;
    }
    return isMemoryEffectFree(op);
  }
  static bool isLocalAllocMemref(Value memref, const llvm::DenseSet<Value> &allocResults) {
    return memref && allocResults.contains(memref);
  }

  Value cloneLoadWithRemappedIndices(Operation *defOp, OpBuilder &builder, ValueMap &remap);
  Value cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap, llvm::DenseSet<Value> &visitingValues,
                     const llvm::DenseSet<Value> &allocResults);
  FuncScanInfo scanFunctionOnce(func::FuncOp func);
  Operation *findForwardableStore(Operation *loadOp, const llvm::DenseSet<Value> &allocResults);
  void collectSiblingCandidates(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
                                SmallVectorImpl<RecomputeCandidate> &candidates,
                                const llvm::DenseSet<Value> &allocResults);
  bool rewriteCandidates(ArrayRef<RecomputeCandidate> candidates, llvm::DenseSet<Value> &rewrittenMemrefs,
                         SmallVectorImpl<Operation *> &toErase, const llvm::DenseSet<Value> &allocResults);
  void eraseDeadLocalStores(ArrayRef<memref::AllocOp> allocs, const llvm::DenseSet<Value> &rewrittenMemrefs);
};

Value ReductionSiblingRecomputePass::cloneLoadWithRemappedIndices(Operation *defOp, OpBuilder &builder,
                                                                  ValueMap &remap) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
    SmallVector<Value> remappedIndices;
    remappedIndices.reserve(loadOp.getIndices().size());
    for (Value index : loadOp.getIndices()) {
      Value remappedIndex = remap.lookup(index);
      if (!remappedIndex) {
        if (auto blockArg = dyn_cast<BlockArgument>(index)) {
          remappedIndex = blockArg;
        } else {
          remappedIndex = index;
        }
      }
      if (!remappedIndex) {
        return {};
      }
      remappedIndices.push_back(remappedIndex);
    }
    auto clonedLoad =
      builder.create<affine::AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(), loadOp.getAffineMap(), remappedIndices);
    return clonedLoad.getResult();
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
    SmallVector<Value> remappedIndices;
    remappedIndices.reserve(loadOp.getIndices().size());
    for (Value index : loadOp.getIndices()) {
      Value remappedIndex = remap.lookup(index);
      if (!remappedIndex) {
        if (auto blockArg = dyn_cast<BlockArgument>(index)) {
          remappedIndex = blockArg;
        } else {
          remappedIndex = index;
        }
      }
      if (!remappedIndex) {
        return {};
      }
      remappedIndices.push_back(remappedIndex);
    }
    auto clonedLoad = builder.create<memref::LoadOp>(loadOp.getLoc(), loadOp.getMemRef(), remappedIndices);
    return clonedLoad.getResult();
  }

  return {};
}

Value ReductionSiblingRecomputePass::cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap,
                                                  llvm::DenseSet<Value> &visitingValues,
                                                  const llvm::DenseSet<Value> &allocResults) {
  if (!value) {
    return {};
  }
  if (auto it = remap.find(value); it != remap.end()) {
    return it->second;
  }
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    remap[value] = blockArg;
    return blockArg;
  }
  if (!visitingValues.insert(value).second) {
    return {};
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !isSupportedCloneOp(defOp)) {
    visitingValues.erase(value);
    return {};
  }

  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
    if (isLocalAllocMemref(loadOp.getMemRef(), allocResults)) {
      Operation *storeOp = findForwardableStore(loadOp, allocResults);
      if (!storeOp) {
        visitingValues.erase(value);
        return {};
      }
      Value cloned = cloneValueAt(CommonUtils::getStoreValue(storeOp), builder, remap, visitingValues, allocResults);
      remap[value] = cloned;
      visitingValues.erase(value);
      return cloned;
    }
    Value clonedLoad = cloneLoadWithRemappedIndices(defOp, builder, remap);
    visitingValues.erase(value);
    if (clonedLoad) {
      remap[value] = clonedLoad;
    }
    return clonedLoad;
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
    if (isLocalAllocMemref(loadOp.getMemRef(), allocResults)) {
      Operation *storeOp = findForwardableStore(loadOp, allocResults);
      if (!storeOp) {
        visitingValues.erase(value);
        return {};
      }
      Value cloned = cloneValueAt(CommonUtils::getStoreValue(storeOp), builder, remap, visitingValues, allocResults);
      remap[value] = cloned;
      visitingValues.erase(value);
      return cloned;
    }
    Value clonedLoad = cloneLoadWithRemappedIndices(defOp, builder, remap);
    visitingValues.erase(value);
    if (clonedLoad) {
      remap[value] = clonedLoad;
    }
    return clonedLoad;
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(defOp->getNumOperands());
  for (Value operand : defOp->getOperands()) {
    Value cloned = cloneValueAt(operand, builder, remap, visitingValues, allocResults);
    if (!cloned) {
      visitingValues.erase(value);
      return {};
    }
    newOperands.push_back(cloned);
  }

  OperationState state(defOp->getLoc(), defOp->getName().getStringRef());
  state.addOperands(newOperands);
  state.addTypes(defOp->getResultTypes());
  state.addAttributes(defOp->getAttrs());
  Operation *clonedOp = Operation::create(state);
  builder.insert(clonedOp);

  for (auto [orig, cloned] : llvm::zip(defOp->getResults(), clonedOp->getResults())) {
    remap[orig] = cloned;
  }
  visitingValues.erase(value);
  return remap.lookup(value);
}

Operation *ReductionSiblingRecomputePass::findForwardableStore(Operation *loadOp,
                                                               const llvm::DenseSet<Value> &allocResults) {
  if (!loadOp || !loadOp->getBlock() || !isLocalAllocMemref(loadOp->getOperand(0), allocResults)) {
    return nullptr;
  }

  for (Operation *cursor = loadOp->getPrevNode(); cursor; cursor = cursor->getPrevNode()) {
    if (isa<affine::AffineStoreOp, memref::StoreOp>(cursor) && sameAccessInBlock(cursor, loadOp)) {
      return cursor;
    }
    if (!isCloneTransparentOp(cursor)) {
      break;
    }
  }
  return nullptr;
}

void ReductionSiblingRecomputePass::collectSiblingCandidates(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
                                                             SmallVectorImpl<RecomputeCandidate> &candidates,
                                                             const llvm::DenseSet<Value> &allocResults) {
  SmallVector<affine::AffineStoreOp> srcStores;
  srcLoop.walk([&](affine::AffineStoreOp storeOp) { srcStores.push_back(storeOp); });

  SmallVector<affine::AffineLoadOp> dstLoads;
  dstLoop.walk([&](affine::AffineLoadOp loadOp) { dstLoads.push_back(loadOp); });

  for (affine::AffineLoadOp loadOp : dstLoads) {
    if (!loadOp || !loadOp->getBlock()) {
      continue;
    }
    if (!isLocalAllocMemref(loadOp.getMemRef(), allocResults)) {
      continue;
    }

    for (affine::AffineStoreOp storeOp : srcStores) {
      if (!storeOp || !storeOp->getBlock()) {
        continue;
      }
      if (sameSiblingAccess(storeOp, srcLoop, loadOp, dstLoop)) {
        candidates.push_back({srcLoop, dstLoop, loadOp, storeOp});
        break;
      }
    }
  }
}

bool ReductionSiblingRecomputePass::rewriteCandidates(ArrayRef<RecomputeCandidate> candidates,
                                                      llvm::DenseSet<Value> &rewrittenMemrefs,
                                                      SmallVectorImpl<Operation *> &toErase,
                                                      const llvm::DenseSet<Value> &allocResults) {
  bool changed = false;
  for (const RecomputeCandidate &candidate : candidates) {
    affine::AffineForOp srcLoop = candidate.srcLoop;
    affine::AffineForOp dstLoop = candidate.dstLoop;
    affine::AffineLoadOp loadOp = candidate.dstLoad;
    affine::AffineStoreOp matchedStore = candidate.srcStore;
    if (!srcLoop || !dstLoop || !loadOp || !matchedStore) {
      continue;
    }

    ValueMap remap;
    remap[srcLoop.getInductionVar()] = dstLoop.getInductionVar();
    OpBuilder builder(loadOp);
    llvm::DenseSet<Value> visitingValues;
    Value cloned = cloneValueAt(matchedStore.getValueToStore(), builder, remap, visitingValues, allocResults);
    if (!cloned) {
      continue;
    }
    loadOp.getResult().replaceAllUsesWith(cloned);
    rewrittenMemrefs.insert(matchedStore.getMemRef());
    toErase.push_back(loadOp);
    changed = true;
  }
  return changed;
}

ReductionSiblingRecomputePass::FuncScanInfo ReductionSiblingRecomputePass::scanFunctionOnce(func::FuncOp func) {
  FuncScanInfo info;
  func.walk([&](Operation *op) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op); forOp && forOp->getParentOp() == func.getOperation()) {
      info.outerLoops.push_back(forOp);
    }
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      info.allocs.push_back(allocOp);
      info.allocResults.insert(allocOp.getResult());
    }
  });
  return info;
}

void ReductionSiblingRecomputePass::eraseDeadLocalStores(ArrayRef<memref::AllocOp> allocs,
                                                         const llvm::DenseSet<Value> &rewrittenMemrefs) {
  for (memref::AllocOp allocOp : allocs) {
    if (!rewrittenMemrefs.contains(allocOp.getResult())) {
      continue;
    }
    bool onlyStores = true;
    SmallVector<Operation *> storeUsers;
    for (Operation *user : allocOp->getUsers()) {
      if (isa<affine::AffineStoreOp, memref::StoreOp>(user)) {
        storeUsers.push_back(user);
        continue;
      }
      onlyStores = false;
      break;
    }
    if (!onlyStores || storeUsers.empty()) {
      continue;
    }
    for (Operation *store : storeUsers) {
      store->erase();
    }
    if (allocOp->use_empty()) {
      allocOp.erase();
    }
  }
}

void ReductionSiblingRecomputePass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (CommonUtils::getOperatorType(func) != OperatorTemplate::Reduction) {
    return;
  }

  SmallVector<Operation *> toErase;
  bool changed = false;
  FuncScanInfo scanInfo = scanFunctionOnce(func);
  llvm::DenseSet<Value> rewrittenMemrefs;
  SmallVector<RecomputeCandidate> candidates;

  for (affine::AffineForOp outer : scanInfo.outerLoops) {
    SmallVector<affine::AffineForOp> childLoops;
    for (Operation &op : outer.getBody()->getOperations()) {
      if (auto child = dyn_cast<affine::AffineForOp>(&op)) {
        childLoops.push_back(child);
      }
    }

    for (size_t i = 0; i < childLoops.size(); ++i) {
      if (!childLoops[i]->hasAttr(kReductionLoopAttr)) {
        continue;
      }
      for (size_t j = i + 1; j < childLoops.size(); ++j) {
        collectSiblingCandidates(childLoops[i], childLoops[j], candidates, scanInfo.allocResults);
      }
    }
  }

  changed |= rewriteCandidates(candidates, rewrittenMemrefs, toErase, scanInfo.allocResults);

  llvm::SmallVector<Operation *, 4> dedup;
  for (Operation *op : toErase) {
    if (op && llvm::find(dedup, op) == dedup.end()) {
      dedup.push_back(op);
      op->erase();
    }
  }

  if (changed) {
    eraseDeadLocalStores(scanInfo.allocs, rewrittenMemrefs);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createReductionSiblingRecomputePass() {
  return std::make_unique<ReductionSiblingRecomputePass>();
}

}  // namespace mlir
