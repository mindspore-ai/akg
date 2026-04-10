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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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

// Collects nested affine.for ops from op up to (not including) outerLoop, outermost first.
static SmallVector<affine::AffineForOp> getNestedLoopChain(Operation *op, Operation *outerLoop) {
  SmallVector<affine::AffineForOp> chain;
  for (Operation *parent = op->getParentOp(); parent && parent != outerLoop; parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(parent)) {
      chain.push_back(forOp);
    }
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

// Returns true if two affine.for loops have the same iteration bounds and step.
static bool sameLoopBounds(affine::AffineForOp a, affine::AffineForOp b) {
  return a.getLowerBoundMap() == b.getLowerBoundMap() && a.getUpperBoundMap() == b.getUpperBoundMap() &&
         a.getStep() == b.getStep() && llvm::equal(a.getLowerBoundOperands(), b.getLowerBoundOperands()) &&
         llvm::equal(a.getUpperBoundOperands(), b.getUpperBoundOperands());
}

struct ReductionSiblingRecomputePass : public impl::ReductionSiblingRecomputeBase<ReductionSiblingRecomputePass> {
 public:
  void runOnOperation() override;

 private:
  using ValueMap = llvm::DenseMap<Value, Value>;

  struct TraceEntry {
    affine::AffineForOp writerLoop;
    affine::AffineStoreOp store;
  };

  struct FuncScanInfo {
    SmallVector<memref::AllocOp> allocs;
    llvm::DenseMap<Value, TraceEntry> memrefTraceMap;
    llvm::DenseSet<Value> reduceOutputMemrefs;
    static bool isLocalAlloc(Value memref) { return memref && memref.getDefiningOp<memref::AllocOp>(); }
  };

  FuncScanInfo scanInfo;

  void printScanInfo(llvm::raw_ostream &os);
  void dumpScanInfo() { printScanInfo(llvm::dbgs()); }

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
  Operation *findForwardableStore(Operation *loadOp);

  // Checks if store in writerLoop and load access the same element, updating remap for IV mapping.
  bool matchCrossLoopAccess(affine::AffineStoreOp storeOp, affine::AffineForOp writerLoop, affine::AffineLoadOp loadOp,
                            affine::AffineForOp dstLoop, ValueMap &remap);

  // Cloning with transitive cross-loop tracing.
  Value cloneLoadValue(Operation *defOp, OpBuilder &builder, ValueMap &remap, llvm::DenseSet<Value> &visitingValues,
                       affine::AffineForOp dstLoop);
  Value cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap, llvm::DenseSet<Value> &visitingValues,
                     affine::AffineForOp dstLoop);

  // Emits a load with remapped indices.
  Value emitRemappedLoad(Operation *defOp, OpBuilder &builder, ValueMap &remap);

  // Per-loop processing: single walk collects loads + stores, rewrites loads, then updates trace map.
  void processChildLoop(affine::AffineForOp childLoop, llvm::DenseSet<Value> &rewrittenMemrefs,
                        SmallVectorImpl<Operation *> &toErase);
  // Processes direct child affine.for loops of a block as siblings.
  void processSiblingLoops(Block &block, llvm::DenseSet<Value> &rewrittenMemrefs,
                           SmallVectorImpl<Operation *> &toErase);

  // Cleanup.
  void eraseDeadLocalStores(const llvm::DenseSet<Value> &rewrittenMemrefs);
};

void ReductionSiblingRecomputePass::printScanInfo(llvm::raw_ostream &os) {
  os << "=== ReductionSiblingRecompute ScanInfo ===\n";

  os << "Allocs (" << scanInfo.allocs.size() << "):\n";
  for (auto allocOp : scanInfo.allocs) {
    os << "  ";
    allocOp.getResult().print(os);
    os << " : " << allocOp.getResult().getType() << "\n";
  }

  os << "Reduce output memrefs (" << scanInfo.reduceOutputMemrefs.size() << "):\n";
  for (Value memref : scanInfo.reduceOutputMemrefs) {
    os << "  ";
    memref.print(os);
    os << "\n";
  }

  os << "Memref trace map (" << scanInfo.memrefTraceMap.size() << "):\n";
  for (auto &[memref, entry] : scanInfo.memrefTraceMap) {
    os << "  ";
    memref.print(os);
    os << "\n    writerLoop: ";
    entry.writerLoop.getLoc().print(os);
    os << "\n    store: ";
    entry.store->print(os);
    os << "\n";
  }

  os << "==========================================\n";
}

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

// Checks if store in writerLoop and load access the same element.
// Handles both direct (load in dstLoop) and transitive (load in intermediate loop) cases.
// Updates remap to map writerLoop's IV and load's IV to dstLoop's IV.
bool ReductionSiblingRecomputePass::matchCrossLoopAccess(affine::AffineStoreOp storeOp, affine::AffineForOp writerLoop,
                                                         affine::AffineLoadOp loadOp, affine::AffineForOp dstLoop,
                                                         ValueMap &remap) {
  if (storeOp.getMemRef() != loadOp.getMemRef() || storeOp.getAffineMap() != loadOp.getAffineMap()) {
    return false;
  }
  auto storeIndices = storeOp.getIndices();
  auto loadIndices = loadOp.getIndices();
  if (storeIndices.size() != loadIndices.size()) {
    return false;
  }
  // Collect nested loop chains for inner loop IV matching.
  auto storeNested = getNestedLoopChain(storeOp, writerLoop);
  auto loadNested = getNestedLoopChain(loadOp, dstLoop);

  for (auto [storeIdx, loadIdx] : llvm::zip(storeIndices, loadIndices)) {
    if (storeIdx == loadIdx) {
      continue;
    }
    // The differing index is the writerLoop's IV.
    if (storeIdx == writerLoop.getInductionVar()) {
      // Map writerLoop's IV to dstLoop's IV.
      remap[writerLoop.getInductionVar()] = dstLoop.getInductionVar();
      // If loadIdx is an intermediate loop's IV (not dstLoop's), also remap it.
      if (loadIdx != dstLoop.getInductionVar()) {
        remap[loadIdx] = dstLoop.getInductionVar();
      }
      continue;
    }
    // Try matching inner loop IVs at corresponding nesting depths.
    bool matched = false;
    for (auto [sLoop, lLoop] : llvm::zip(storeNested, loadNested)) {
      if (storeIdx == sLoop.getInductionVar()) {
        if (sameLoopBounds(sLoop, lLoop)) {
          remap[storeIdx] = lLoop.getInductionVar();
          matched = true;
        }
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

// Emits a load op with remapped indices at the builder's insertion point.
Value ReductionSiblingRecomputePass::emitRemappedLoad(Operation *defOp, OpBuilder &builder, ValueMap &remap) {
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

// Clones a load op with transitive cross-loop tracing.
// For non-local alloc or reduce output memrefs, emits a load with remapped indices.
// For local alloc non-reduce memrefs, tries same-block forwarding first, then cross-loop tracing via memrefTraceMap.
Value ReductionSiblingRecomputePass::cloneLoadValue(Operation *defOp, OpBuilder &builder, ValueMap &remap,
                                                    llvm::DenseSet<Value> &visitingValues,
                                                    affine::AffineForOp dstLoop) {
  Value loadMemref;
  if (auto affineLoad = dyn_cast<affine::AffineLoadOp>(defOp)) {
    loadMemref = affineLoad.getMemRef();
  } else if (auto memrefLoad = dyn_cast<memref::LoadOp>(defOp)) {
    loadMemref = memrefLoad.getMemRef();
  } else {
    return {};
  }

  // Non-local alloc or reduce output: emit load with remapped indices (keep as-is).
  if (!FuncScanInfo::isLocalAlloc(loadMemref) || scanInfo.reduceOutputMemrefs.contains(loadMemref)) {
    return emitRemappedLoad(defOp, builder, remap);
  }

  // Same-block store forwarding (within the source loop body).
  if (auto *storeOp = findForwardableStore(defOp)) {
    return cloneValueAt(CommonUtils::getStoreValue(storeOp), builder, remap, visitingValues, dstLoop);
  }

  // Cross-loop store forwarding via precomputed trace map.
  auto it = scanInfo.memrefTraceMap.find(loadMemref);
  if (it != scanInfo.memrefTraceMap.end()) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(defOp)) {
      TraceEntry &entry = it->second;
      if (matchCrossLoopAccess(entry.store, entry.writerLoop, loadOp, dstLoop, remap)) {
        return cloneValueAt(entry.store.getValueToStore(), builder, remap, visitingValues, dstLoop);
      }
    }
  }

  // Fallback: emit load with remapped indices.
  return emitRemappedLoad(defOp, builder, remap);
}

// Recursively clones value and its defining op at builder, with IV remapping.
// Handles transitive cross-loop dependencies via cloneLoadValue.
// Returns {} on cycle or unsupported op.
Value ReductionSiblingRecomputePass::cloneValueAt(Value value, OpBuilder &builder, ValueMap &remap,
                                                  llvm::DenseSet<Value> &visitingValues, affine::AffineForOp dstLoop) {
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

  // Value defined at the outer loop body level or above: directly accessible from dstLoop.
  Operation *outerLoop = dstLoop->getParentOp();
  if (defOp->getParentOp() == outerLoop || !outerLoop->isProperAncestor(defOp)) {
    remap[value] = value;
    return value;
  }

  if (isa<affine::AffineLoadOp, memref::LoadOp>(defOp)) {
    Value clonedLoad = cloneLoadValue(defOp, builder, remap, visitingValues, dstLoop);
    if (clonedLoad) {
      remap[value] = clonedLoad;
    }
    return clonedLoad;
  }

  SmallVector<Value> newOperands;
  newOperands.reserve(defOp->getNumOperands());
  for (Value operand : defOp->getOperands()) {
    Value cloned = cloneValueAt(operand, builder, remap, visitingValues, dstLoop);
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

// Collects candidate loads and stores from childLoop (including nested inner loops),
// rewrites loads, then updates trace map.
void ReductionSiblingRecomputePass::processChildLoop(affine::AffineForOp childLoop,
                                                     llvm::DenseSet<Value> &rewrittenMemrefs,
                                                     SmallVectorImpl<Operation *> &toErase) {
  SmallVector<affine::AffineLoadOp> candidateLoads;
  SmallVector<affine::AffineStoreOp> traceStores;

  // Walk all ops including nested inner loops to find candidate loads and stores.
  childLoop.walk([&](Operation *opPtr) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(opPtr)) {
      Value memref = loadOp.getMemRef();
      if (FuncScanInfo::isLocalAlloc(memref) && !scanInfo.reduceOutputMemrefs.contains(memref) &&
          scanInfo.memrefTraceMap.count(memref)) {
        candidateLoads.push_back(loadOp);
      }
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(opPtr)) {
      Value memref = storeOp.getMemRef();
      if (FuncScanInfo::isLocalAlloc(memref) && !scanInfo.reduceOutputMemrefs.contains(memref)) {
        traceStores.push_back(storeOp);
      }
    }
  });

  // Phase 1: rewrite loads using trace map from PREVIOUS siblings.
  for (affine::AffineLoadOp loadOp : candidateLoads) {
    auto it = scanInfo.memrefTraceMap.find(loadOp.getMemRef());
    if (it == scanInfo.memrefTraceMap.end()) {
      continue;
    }
    TraceEntry &entry = it->second;

    ValueMap remap;
    if (!matchCrossLoopAccess(entry.store, entry.writerLoop, loadOp, childLoop, remap)) {
      continue;
    }

    OpBuilder builder(loadOp);
    llvm::DenseSet<Value> visitingValues;
    Value cloned = cloneValueAt(entry.store.getValueToStore(), builder, remap, visitingValues, childLoop);
    if (!cloned) {
      continue;
    }

    loadOp.getResult().replaceAllUsesWith(cloned);
    rewrittenMemrefs.insert(loadOp.getMemRef());
    toErase.push_back(loadOp);
  }

  // Phase 2: update trace map with this loop's stores for FUTURE siblings.
  for (affine::AffineStoreOp storeOp : traceStores) {
    scanInfo.memrefTraceMap[storeOp.getMemRef()] = {childLoop, storeOp};
  }
}

// Recursively processes sibling loops at all nesting levels.
// At each level: clears trace map, processes siblings, then recurses into each child loop.
void ReductionSiblingRecomputePass::processSiblingLoops(Block &block, llvm::DenseSet<Value> &rewrittenMemrefs,
                                                        SmallVectorImpl<Operation *> &toErase) {
  SmallVector<affine::AffineForOp> siblings;
  for (Operation &op : block) {
    if (auto forOp = dyn_cast<affine::AffineForOp>(&op)) {
      siblings.push_back(forOp);
    }
  }

  // Only process at this level if there are 2+ siblings; otherwise no cross-loop deps to resolve.
  if (siblings.size() > 1) {
    auto savedTraceMap = std::move(scanInfo.memrefTraceMap);
    scanInfo.memrefTraceMap.clear();

    for (affine::AffineForOp loop : siblings) {
      processChildLoop(loop, rewrittenMemrefs, toErase);
    }

    scanInfo.memrefTraceMap = std::move(savedTraceMap);
  }

  // Recurse into each child loop's body.
  for (affine::AffineForOp loop : siblings) {
    processSiblingLoops(*loop.getBody(), rewrittenMemrefs, toErase);
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

  scanInfo.allocs.clear();
  scanInfo.memrefTraceMap.clear();
  scanInfo.reduceOutputMemrefs.clear();

  // Collect allocs and identify reduce output memrefs across the entire function.
  func.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
      scanInfo.allocs.push_back(allocOp);
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      if (storeOp->hasAttr(kReductionInitAttr)) {
        scanInfo.reduceOutputMemrefs.insert(storeOp.getMemRef());
      } else {
        Operation *defOp = storeOp.getValueToStore().getDefiningOp();
        if (defOp && defOp->hasAttr(kReductionAxesStr)) {
          scanInfo.reduceOutputMemrefs.insert(storeOp.getMemRef());
        }
      }
    }
  });

  LLVM_DEBUG(dumpScanInfo());

  llvm::DenseSet<Value> rewrittenMemrefs;
  SmallVector<Operation *> toErase;

  // Recursively process sibling loops at all nesting levels.
  processSiblingLoops(func.getBody().front(), rewrittenMemrefs, toErase);

  if (!toErase.empty()) {
    llvm::SmallPtrSet<Operation *, 4> seen;
    for (Operation *op : toErase) {
      if (op && seen.insert(op).second) {
        op->erase();
      }
    }
    eraseDeadLocalStores(rewrittenMemrefs);
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createReductionSiblingRecomputePass() {
  return std::make_unique<ReductionSiblingRecomputePass>();
}

}  // namespace mlir
