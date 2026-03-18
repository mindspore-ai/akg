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
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
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
    SmallVector<memref::AllocOp> allocs;
    static bool isLocalAlloc(Value memref) { return memref && memref.getDefiningOp<memref::AllocOp>(); }
  };
  struct RecomputeCandidate {
    affine::AffineForOp srcLoop;
    affine::AffineForOp dstLoop;
    affine::AffineLoadOp dstLoad;
    affine::AffineStoreOp srcStore;
  };
  FuncScanInfo scanInfo = {};

  // Returns true if store and load target the same memref and indices (same block).
  static bool sameAccessInBlock(Operation *storeOp, Operation *loadOp) {
    if (CommonUtils::getStoreMemref(storeOp) != loadOp->getOperand(0)) {
      return false;
    }
    auto storeIndices = CommonUtils::getStoreLoadIndices(storeOp);
    auto loadIndices = CommonUtils::getStoreLoadIndices(loadOp);
    if (storeIndices.size() != loadIndices.size() || !llvm::equal(storeIndices, loadIndices)) {
      return false;
    }
    auto affineStore = dyn_cast<affine::AffineStoreOp>(storeOp);
    auto affineLoad = dyn_cast<affine::AffineLoadOp>(loadOp);
    if (affineStore || affineLoad) {
      return affineStore && affineLoad && affineStore.getAffineMap() == affineLoad.getAffineMap();
    }
    return true;
  }

  // Returns true if store in srcLoop and load in dstLoop access the same element (IVs may differ).
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
      if (storeIdx == loadIdx || (storeIdx == srcLoop.getInductionVar() && loadIdx == dstLoop.getInductionVar())) {
        continue;
      }
      return false;
    }
    return true;
  }

  // Returns true if op is safe to clone (no regions; loads optional; otherwise memory-effect free).
  static bool isSupportedCloneOp(Operation *op, bool allowLoads = true) {
    if (!op || op->getNumRegions() != 0) {
      return false;
    }
    if (allowLoads && isa<affine::AffineLoadOp, memref::LoadOp>(op)) {
      return op->getNumResults() > 0;
    }
    return isMemoryEffectFree(op);
  }

  // Lookup value in remap; returns original if not mapped.
  static Value getRemappedValue(Value value, ValueMap &remap) {
    if (Value mapped = remap.lookup(value)) {
      return mapped;
    }
    return value;
  }

  SmallVector<Value> remapIndices(ValueRange indices, ValueMap &remap);
  Value cloneLoadValue(Operation *defOp, OpBuilder &builder, ValueMap &remap, llvm::DenseSet<Value> &visitingValues);
  Value cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap, llvm::DenseSet<Value> &visitingValues);
  Operation *findForwardableStore(Operation *loadOp);
  void collectSiblingCandidates(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
                                SmallVectorImpl<RecomputeCandidate> &candidates);
  void collectCandidates(func::FuncOp func, SmallVectorImpl<RecomputeCandidate> &candidates);
  bool rewriteCandidates(ArrayRef<RecomputeCandidate> candidates, llvm::DenseSet<Value> &rewrittenMemrefs,
                         SmallVectorImpl<Operation *> &toErase);
  void eraseOpsDedup(ArrayRef<Operation *> ops);
  void eraseDeadLocalStores(const llvm::DenseSet<Value> &rewrittenMemrefs);
};

// Remaps index values using remap; returns empty on failure.
SmallVector<Value> ReductionSiblingRecomputePass::remapIndices(ValueRange indices, ValueMap &remap) {
  SmallVector<Value> remappedIndices;
  remappedIndices.reserve(indices.size());
  for (Value index : indices) {
    Value remappedIndex = getRemappedValue(index, remap);
    if (!remappedIndex) {
      return {};
    }
    remappedIndices.push_back(remappedIndex);
  }
  return remappedIndices;
}

// Clones a load op: forwards from local alloc store or creates load with remapped indices.
Value ReductionSiblingRecomputePass::cloneLoadValue(Operation *defOp, OpBuilder &builder, ValueMap &remap,
                                                    llvm::DenseSet<Value> &visitingValues) {
  Value loadMemref;
  if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(defOp)) {
    loadMemref = affineLoad.getMemRef();
  } else if (auto memrefLoad = dyn_cast<memref::LoadOp>(defOp)) {
    loadMemref = memrefLoad.getMemRef();
  } else {
    return {};
  }

  if (FuncScanInfo::isLocalAlloc(loadMemref)) {
    if (auto *storeOp = findForwardableStore(defOp)) {
      return cloneValueAt(CommonUtils::getStoreValue(storeOp), builder, remap, visitingValues);
    }
  }

  auto indices = CommonUtils::getStoreLoadIndices(defOp);
  SmallVector<Value> remappedIndices = remapIndices(indices, remap);
  if (remappedIndices.empty() && !indices.empty()) {
    return {};
  }
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
    return builder
      .create<affine::AffineLoadOp>(loadOp.getLoc(), loadOp.getMemRef(), loadOp.getAffineMap(), remappedIndices)
      .getResult();
  }
  if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
    return builder.create<memref::LoadOp>(loadOp.getLoc(), loadOp.getMemRef(), remappedIndices).getResult();
  }
  return {};
}

// Recursively clones value and its defining op at builder, with IV remapping.
// Returns {} on cycle or unsupported op.
Value ReductionSiblingRecomputePass::cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap,
                                                  llvm::DenseSet<Value> &visitingValues) {
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
  auto guard = llvm::make_scope_exit([&] { visitingValues.erase(value); });

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !isSupportedCloneOp(defOp)) {
    return {};
  }

  if (isa<affine::AffineLoadOp, memref::LoadOp>(defOp)) {
    Value clonedLoad = cloneLoadValue(defOp, builder, remap, visitingValues);
    if (clonedLoad) {
      remap[value] = clonedLoad;
    }
    return clonedLoad;
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(defOp->getNumOperands());
  for (Value operand : defOp->getOperands()) {
    Value cloned = cloneValueAt(operand, builder, remap, visitingValues);
    if (!cloned) {
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
  return remap.lookup(value);
}

// Finds the dominating store in the same block that writes the same location as loadOp (local alloc only).
Operation *ReductionSiblingRecomputePass::findForwardableStore(Operation *loadOp) {
  if (!loadOp || !loadOp->getBlock() || !FuncScanInfo::isLocalAlloc(loadOp->getOperand(0))) {
    return nullptr;
  }

  for (Operation *cursor = loadOp->getPrevNode(); cursor; cursor = cursor->getPrevNode()) {
    if (isa<affine::AffineStoreOp, memref::StoreOp>(cursor) && sameAccessInBlock(cursor, loadOp)) {
      return cursor;
    }
    if (!isSupportedCloneOp(cursor, /*allowLoads=*/false)) {
      break;
    }
  }
  return nullptr;
}

// Collects (srcLoop, dstLoop, load, store) where dstLoop loads from local buffer written by srcLoop (same access).
void ReductionSiblingRecomputePass::collectSiblingCandidates(affine::AffineForOp srcLoop, affine::AffineForOp dstLoop,
                                                             SmallVectorImpl<RecomputeCandidate> &candidates) {
  SmallVector<affine::AffineStoreOp> srcStores;
  srcLoop.walk([&](affine::AffineStoreOp storeOp) { srcStores.push_back(storeOp); });

  SmallVector<affine::AffineLoadOp> dstLoads;
  dstLoop.walk([&](affine::AffineLoadOp loadOp) { dstLoads.push_back(loadOp); });

  for (affine::AffineLoadOp loadOp : dstLoads) {
    if (!FuncScanInfo::isLocalAlloc(loadOp.getMemRef())) {
      continue;
    }
    auto it = llvm::find_if(
      srcStores, [&](affine::AffineStoreOp storeOp) { return sameSiblingAccess(storeOp, srcLoop, loadOp, dstLoop); });
    if (it != srcStores.end()) {
      candidates.push_back({srcLoop, dstLoop, loadOp, *it});
    }
  }
}

// Replaces candidate loads with recomputed values; records rewritten memrefs and ops to erase.
bool ReductionSiblingRecomputePass::rewriteCandidates(ArrayRef<RecomputeCandidate> candidates,
                                                      llvm::DenseSet<Value> &rewrittenMemrefs,
                                                      SmallVectorImpl<Operation *> &toErase) {
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
    Value cloned = cloneValueAt(matchedStore.getValueToStore(), builder, remap, visitingValues);
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

// collects allocs and (reduction loop, sibling loop) load/store candidates for recompute.
void ReductionSiblingRecomputePass::collectCandidates(func::FuncOp func,
                                                      SmallVectorImpl<RecomputeCandidate> &candidates) {
  func.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      scanInfo.allocs.push_back(allocOp);
      return;
    }
    auto outerLoop = dyn_cast<affine::AffineForOp>(op);
    if (!outerLoop || outerLoop->getParentOp() != func.getOperation()) {
      return;
    }
    SmallVector<affine::AffineForOp> childLoops;
    for (Operation &childOp : outerLoop.getBody()->getOperations()) {
      if (auto child = dyn_cast<affine::AffineForOp>(&childOp)) {
        childLoops.push_back(child);
      }
    }
    for (size_t i = 0; i < childLoops.size(); ++i) {
      if (!childLoops[i]->hasAttr(kReductionLoopAttr)) {
        continue;
      }
      for (size_t j = i + 1; j < childLoops.size(); ++j) {
        collectSiblingCandidates(childLoops[i], childLoops[j], candidates);
      }
    }
  });
}

// Erases each op once (dedup by pointer).
void ReductionSiblingRecomputePass::eraseOpsDedup(ArrayRef<Operation *> ops) {
  llvm::SmallPtrSet<Operation *, 4> seen;
  for (Operation *op : ops) {
    if (op && seen.insert(op).second) {
      op->erase();
    }
  }
}

// Removes stores (and alloc if unused) for rewritten local memrefs that now have only stores.
// First forwards any intra-block store-to-load pairs so the loads can be eliminated.
void ReductionSiblingRecomputePass::eraseDeadLocalStores(const llvm::DenseSet<Value> &rewrittenMemrefs) {
  for (memref::AllocOp allocOp : scanInfo.allocs) {
    if (!rewrittenMemrefs.contains(allocOp.getResult())) {
      continue;
    }
    SmallVector<Operation *> loadsToErase;
    for (Operation *user : allocOp->getUsers()) {
      if (!isa<affine::AffineLoadOp, memref::LoadOp>(user)) {
        continue;
      }
      if (auto *storeOp = findForwardableStore(user)) {
        user->getResult(0).replaceAllUsesWith(CommonUtils::getStoreValue(storeOp));
        loadsToErase.push_back(user);
      }
    }
    for (Operation *load : loadsToErase) {
      load->erase();
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

  // collects candidates
  SmallVector<RecomputeCandidate> candidates;
  collectCandidates(func, candidates);

  // rewrites loads
  llvm::DenseSet<Value> rewrittenMemrefs;
  SmallVector<Operation *> toErase;
  bool changed = rewriteCandidates(candidates, rewrittenMemrefs, toErase);
  if (changed) {
    // erases ops
    eraseOpsDedup(toErase);
    eraseDeadLocalStores(rewrittenMemrefs);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createReductionSiblingRecomputePass() {
  return std::make_unique<ReductionSiblingRecomputePass>();
}

}  // namespace mlir
