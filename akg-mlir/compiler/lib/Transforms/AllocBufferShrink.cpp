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

#include <algorithm>

#include "akg/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#ifndef GEN_PASS_DECL_ALLOCBUFFERSHRINK
#define GEN_PASS_DECL_ALLOCBUFFERSHRINK
#ifndef GEN_PASS_DEF_ALLOCBUFFERSHRINK
#define GEN_PASS_DEF_ALLOCBUFFERSHRINK
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "alloc-buffer-shrink"

namespace mlir {
namespace {

// ============================================================================
// Constants & Types
// ============================================================================

// Sentinel for AccessInfo::dimMapping: the alloc dimension has no corresponding
// access index (e.g., a size-1 dim absorbed during collapse_shape).
static constexpr int kDimAbsorbed = -1;

struct AccessInfo {
  Operation *op;
  // dimMapping[d] gives the index position in this access op that corresponds
  // to alloc dimension d.  kDimAbsorbed if the dimension was absorbed by a
  // reshape (e.g., a size-1 dim merged in collapse_shape).
  SmallVector<int> dimMapping;
};

// ============================================================================
// Index & Loop Utilities
// ============================================================================

static bool isAccessOp(Operation *op) {
  return isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp, memref::StoreOp>(op);
}

// If `idx` is the induction variable of an affine.for / scf.for, return that
// loop operation; else null.
static Operation *getOwningLoopOp(Value idx) {
  auto blockArg = dyn_cast<BlockArgument>(idx);
  if (!blockArg) return nullptr;
  auto *parentOp = blockArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<affine::AffineForOp>(parentOp))
    return forOp.getInductionVar() == blockArg ? parentOp : nullptr;
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) return forOp.getInductionVar() == blockArg ? parentOp : nullptr;
  return nullptr;
}

static SmallVector<Value> getAccessIndices(Operation *op) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op))
    return SmallVector<Value>(loadOp.getIndices().begin(), loadOp.getIndices().end());
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op))
    return SmallVector<Value>(storeOp.getIndices().begin(), storeOp.getIndices().end());
  if (auto loadOp = dyn_cast<memref::LoadOp>(op))
    return SmallVector<Value>(loadOp.getIndices().begin(), loadOp.getIndices().end());
  if (auto storeOp = dyn_cast<memref::StoreOp>(op))
    return SmallVector<Value>(storeOp.getIndices().begin(), storeOp.getIndices().end());
  return {};
}

static Value getAccessMemRef(Operation *op) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) return loadOp.getMemRef();
  if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) return storeOp.getMemRef();
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) return loadOp.getMemRef();
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) return storeOp.getMemRef();
  return {};
}

// ============================================================================
// Dimension Mapping Through Reshape Ops
//
// When traversing from an alloc through reshape ops (collapse_shape /
// expand_shape) to terminal access ops, we track how each alloc dimension
// maps to the current memref's dimensions.  These helpers compute the new
// mapping after passing through a reshape.
// ============================================================================

// Compute dimension mapping through a collapse_shape.
//
// collapse_shape merges groups of source dimensions into single result
// dimensions.  For each alloc dim that currently maps to a source dim:
//   - Singleton group {d}        → 1-to-1 mapping to group index.
//   - Multi-dim group, size-1 d  → absorbed (index is implicitly 0).
//   - Multi-dim group, d is the only non-unit dim → maps to group index.
//   - Otherwise                  → too complex, return false.
static bool computeCollapseMapping(MemRefType srcType, ArrayRef<ReassociationIndices> reassociation, unsigned allocRank,
                                   ArrayRef<int> currentMapping, SmallVectorImpl<int> &newMapping) {
  newMapping.assign(allocRank, kDimAbsorbed);
  for (unsigned d = 0; d < allocRank; d++) {
    if (currentMapping[d] == kDimAbsorbed) continue;
    int64_t srcDim = currentMapping[d];

    for (unsigned g = 0; g < reassociation.size(); g++) {
      const auto &group = reassociation[g];
      if (llvm::find(group, srcDim) == group.end()) continue;

      if (group.size() == 1) {
        newMapping[d] = static_cast<int>(g);
        break;
      }

      int64_t dimSize = srcType.getDimSize(srcDim);
      if (ShapedType::isDynamic(dimSize)) return false;
      if (dimSize == 1) break;

      for (int64_t otherDim : group) {
        if (otherDim == srcDim) continue;
        int64_t otherSize = srcType.getDimSize(otherDim);
        if (ShapedType::isDynamic(otherSize) || otherSize != 1) return false;
      }
      newMapping[d] = static_cast<int>(g);
      break;
    }
  }
  return true;
}

// Compute dimension mapping through an expand_shape.
//
// expand_shape splits each source dimension into a group of result dimensions.
// For each alloc dim that currently maps to a source dim:
//   - Singleton group {r}              → 1-to-1 mapping.
//   - Multi-dim group, one non-unit r  → maps to that result dim.
//   - Multi-dim group, all unit        → absorbed.
//   - Otherwise                        → too complex, return false.
static bool computeExpandMapping(MemRefType resultType, ArrayRef<ReassociationIndices> reassociation,
                                 unsigned allocRank, ArrayRef<int> currentMapping, SmallVectorImpl<int> &newMapping) {
  newMapping.assign(allocRank, kDimAbsorbed);
  for (unsigned d = 0; d < allocRank; d++) {
    if (currentMapping[d] == kDimAbsorbed) continue;
    int64_t srcDim = currentMapping[d];
    if (srcDim >= static_cast<int64_t>(reassociation.size())) return false;

    const auto &group = reassociation[srcDim];
    if (group.size() == 1) {
      newMapping[d] = static_cast<int>(group[0]);
      continue;
    }

    int nonUnitIdx = kDimAbsorbed;
    for (int64_t resultDim : group) {
      int64_t dimSize = resultType.getDimSize(resultDim);
      if (ShapedType::isDynamic(dimSize)) return false;
      if (dimSize != 1) {
        if (nonUnitIdx != kDimAbsorbed) return false;
        nonUnitIdx = static_cast<int>(resultDim);
      }
    }
    newMapping[d] = nonUnitIdx;
  }
  return true;
}

// ============================================================================
// View Chain Traversal
// ============================================================================

// Recursively trace through view-like ops to collect terminal access ops
// (with their dimension mappings) and intermediate view ops.
//
// Supported view ops: SubViewOp (same-rank, unit-stride), CollapseShapeOp,
// ExpandShapeOp, MemorySpaceCastOp.  Returns false on unsupported users.
static bool collectAccessOpsRecursive(Value memref, unsigned allocRank, SmallVector<int> currentMapping,
                                      SmallVectorImpl<AccessInfo> &accessInfos, SmallVectorImpl<Operation *> &viewOps) {
  for (auto *user : memref.getUsers()) {
    if (isAccessOp(user)) {
      accessInfos.push_back({user, SmallVector<int>(currentMapping)});
      continue;
    }

    SmallVector<int> newMapping;
    Value nextValue;

    if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
      auto sourceRank = subviewOp.getSourceType().getRank();
      auto targetRank = subviewOp.getType().getRank();
      if (sourceRank != targetRank || targetRank != allocRank) return false;
      auto strides = subviewOp.getStaticStrides();
      if (!llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
        return false;
      }
      newMapping = SmallVector<int>(currentMapping);
      nextValue = subviewOp.getResult();

    } else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(user)) {
      if (!computeCollapseMapping(collapseOp.getSrcType(), collapseOp.getReassociationIndices(), allocRank,
                                  currentMapping, newMapping))
        return false;
      nextValue = collapseOp.getResult();

    } else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(user)) {
      if (!computeExpandMapping(expandOp.getResultType(), expandOp.getReassociationIndices(), allocRank, currentMapping,
                                newMapping))
        return false;
      nextValue = expandOp.getResult();

    } else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(user)) {
      newMapping = SmallVector<int>(currentMapping);
      nextValue = castOp.getResult();

    } else {
      return false;
    }

    viewOps.push_back(user);
    if (!collectAccessOpsRecursive(nextValue, allocRank, newMapping, accessInfos, viewOps)) return false;
  }
  return true;
}

// ============================================================================
// SymShapeAttr Update
// ============================================================================

// Update SymShapeAttr inside a memory-space dictionary: shrunk dimensions
// get their symbolic name replaced with "1".
static Attribute updateMemSpaceForShrink(Attribute memorySpace, ArrayRef<bool> shrinkDims, unsigned rank,
                                         MLIRContext *context) {
  if (!memorySpace) return memorySpace;
  auto dictAttr = dyn_cast<DictionaryAttr>(memorySpace);
  if (!dictAttr) return memorySpace;
  auto symShapeAttr = dictAttr.getAs<ArrayAttr>("SymShapeAttr");
  if (!symShapeAttr || symShapeAttr.size() != rank) return memorySpace;

  SmallVector<Attribute> newSymShapes;
  for (unsigned d = 0; d < rank; d++)
    newSymShapes.push_back(shrinkDims[d] ? StringAttr::get(context, "1") : symShapeAttr[d]);

  SmallVector<NamedAttribute> entries;
  for (auto entry : dictAttr) {
    if (entry.getName().getValue() == "SymShapeAttr")
      entries.push_back({entry.getName(), ArrayAttr::get(context, newSymShapes)});
    else
      entries.push_back(entry);
  }
  return DictionaryAttr::get(context, entries);
}

// ============================================================================
// AllocBufferShrinkPass
// ============================================================================

struct AllocBufferShrinkPass : public AllocBufferShrinkBase<AllocBufferShrinkPass> {
 public:
  void runOnOperation() override;

 private:
  // Per-alloc state, reset at the beginning of each processAlloc() call.
  unsigned rank = 0;
  SmallVector<bool> shrinkDims;
  SmallVector<Operation *> dimLoops;
  SmallVector<AccessInfo> accessInfos;
  SmallVector<Operation *> viewOps;
  DenseMap<Value, Value> replacementMap;
  Value zeroIdx;

  // --- Analysis ---
  bool processAlloc(memref::AllocOp allocOp);
  bool allAccessMapsAreIdentity();
  Operation *findCommonLoopForDim(unsigned d);
  void analyzeShrinkableDims(MemRefType memrefType);
  void validateSubviewOffsets();

  // --- Transformation ---
  memref::AllocOp createShrunkAlloc(memref::AllocOp allocOp, MemRefType memrefType);
  void rebuildViewChain(memref::AllocOp oldAlloc, memref::AllocOp newAlloc);
  void rebuildSubView(memref::SubViewOp subviewOp);
  void rebuildCollapseShape(memref::CollapseShapeOp collapseOp);
  void rebuildExpandShape(memref::ExpandShapeOp expandOp);
  void rebuildMemorySpaceCast(memref::MemorySpaceCastOp castOp);
  void rewriteAccessOps();
  void eraseOldOps(memref::AllocOp allocOp);
};

// --- Analysis ---------------------------------------------------------------

bool AllocBufferShrinkPass::allAccessMapsAreIdentity() {
  for (auto &info : accessInfos) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(info.op)) {
      if (!loadOp.getAffineMap().isIdentity()) return false;
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(info.op)) {
      if (!storeOp.getAffineMap().isIdentity()) return false;
    }
  }
  return true;
}

// Find the unique loop whose induction variable every access op uses to index
// alloc dimension `d`.  Absorbed accesses (dimMapping[d] < 0) are skipped in
// IV matching but still required to live inside the common loop.  Returns null
// if accesses disagree, use a non-IV index, or any access lives outside.
Operation *AllocBufferShrinkPass::findCommonLoopForDim(unsigned d) {
  Operation *commonLoop = nullptr;
  bool allAbsorbed = true;

  for (auto &info : accessInfos) {
    int accessDim = info.dimMapping[d];
    if (accessDim < 0) continue;
    allAbsorbed = false;

    auto indices = getAccessIndices(info.op);
    if (accessDim >= static_cast<int>(indices.size())) return nullptr;
    auto *loopOp = getOwningLoopOp(indices[accessDim]);
    if (!loopOp) return nullptr;
    if (!commonLoop)
      commonLoop = loopOp;
    else if (commonLoop != loopOp)
      return nullptr;
  }

  if (allAbsorbed || !commonLoop) return nullptr;

  for (auto &info : accessInfos) {
    if (!commonLoop->isAncestor(info.op)) return nullptr;
  }
  return commonLoop;
}

void AllocBufferShrinkPass::analyzeShrinkableDims(MemRefType memrefType) {
  auto shape = memrefType.getShape();
  shrinkDims.assign(rank, false);
  dimLoops.assign(rank, nullptr);

  for (unsigned d = 0; d < rank; d++) {
    if (ShapedType::isDynamic(shape[d]) || shape[d] <= 1) continue;
    dimLoops[d] = findCommonLoopForDim(d);
    shrinkDims[d] = (dimLoops[d] != nullptr);
  }
}

// Revoke shrinkability for dimensions where a SubViewOp creates a partial
// alias (non-zero offset or size < source extent).  Also revokes any
// dimension whose controlling loop is nested inside the partial dimension's
// loop, to preserve cross-iteration data dependencies.
void AllocBufferShrinkPass::validateSubviewOffsets() {
  for (auto *op : viewOps) {
    auto subviewOp = dyn_cast<memref::SubViewOp>(op);
    if (!subviewOp) continue;

    auto offsets = subviewOp.getStaticOffsets();
    auto sizes = subviewOp.getStaticSizes();
    auto sourceType = subviewOp.getSourceType();

    for (unsigned j = 0; j < rank; j++) {
      bool hasNonZeroOffset = (offsets[j] != 0);
      bool hasPartialSize = (!ShapedType::isDynamic(sizes[j]) && sizes[j] < sourceType.getDimSize(j));
      if (!hasNonZeroOffset && !hasPartialSize) continue;

      shrinkDims[j] = false;

      if (!dimLoops[j]) continue;
      for (unsigned d = 0; d < rank; d++) {
        if (shrinkDims[d] && dimLoops[d] && dimLoops[j]->isAncestor(dimLoops[d])) shrinkDims[d] = false;
      }
    }
  }
}

// --- Transformation ---------------------------------------------------------

memref::AllocOp AllocBufferShrinkPass::createShrunkAlloc(memref::AllocOp allocOp, MemRefType memrefType) {
  auto *context = &getContext();
  auto shape = memrefType.getShape();

  SmallVector<int64_t> newShape;
  for (unsigned d = 0; d < rank; d++) newShape.push_back(shrinkDims[d] ? 1 : shape[d]);

  auto newMemorySpace = updateMemSpaceForShrink(memrefType.getMemorySpace(), shrinkDims, rank, context);
  auto newType = MemRefType::get(newShape, memrefType.getElementType(), memrefType.getLayout(), newMemorySpace);

  OpBuilder builder(allocOp);
  auto newAlloc = builder.create<memref::AllocOp>(allocOp.getLoc(), newType);
  if (auto alignAttr = allocOp.getAlignmentAttr()) newAlloc->setAttr("alignment", alignAttr);
  return newAlloc;
}

void AllocBufferShrinkPass::rebuildSubView(memref::SubViewOp subviewOp) {
  Value newSource = replacementMap.lookup(subviewOp.getSource());
  if (!newSource) return;

  SmallVector<OpFoldResult> newOffsets = subviewOp.getMixedOffsets();
  SmallVector<OpFoldResult> newSizes = subviewOp.getMixedSizes();
  SmallVector<OpFoldResult> newStrides = subviewOp.getMixedStrides();
  for (unsigned d = 0; d < rank; d++) {
    if (shrinkDims[d]) newSizes[d] = OpBuilder(subviewOp).getI64IntegerAttr(1);
  }

  OpBuilder builder(subviewOp);
  auto newSubview = builder.create<memref::SubViewOp>(subviewOp.getLoc(), newSource, newOffsets, newSizes, newStrides);
  replacementMap[subviewOp.getResult()] = newSubview.getResult();
}

void AllocBufferShrinkPass::rebuildCollapseShape(memref::CollapseShapeOp collapseOp) {
  Value newSource = replacementMap.lookup(collapseOp.getSrc());
  if (!newSource) return;

  OpBuilder builder(collapseOp);
  auto newCollapse =
    builder.create<memref::CollapseShapeOp>(collapseOp.getLoc(), newSource, collapseOp.getReassociationIndices());
  replacementMap[collapseOp.getResult()] = newCollapse.getResult();
}

void AllocBufferShrinkPass::rebuildExpandShape(memref::ExpandShapeOp expandOp) {
  Value newSource = replacementMap.lookup(expandOp.getSrc());
  if (!newSource) return;

  auto newSourceType = dyn_cast<MemRefType>(newSource.getType());
  auto oldResultType = expandOp.getResultType();
  auto reassociation = expandOp.getReassociationIndices();

  SmallVector<int64_t> newOutputShape(oldResultType.getShape().begin(), oldResultType.getShape().end());
  for (unsigned g = 0; g < reassociation.size(); g++) {
    int64_t newSrcSize = newSourceType.getDimSize(g);
    int64_t oldSrcSize = expandOp.getSrcType().getDimSize(g);
    if (newSrcSize == oldSrcSize) continue;
    if (newSrcSize == 1) {
      for (int64_t resultDim : reassociation[g]) newOutputShape[resultDim] = 1;
    }
  }

  OpBuilder builder(expandOp);
  SmallVector<OpFoldResult> outputShapeOfr;
  std::transform(newOutputShape.begin(), newOutputShape.end(), std::back_inserter(outputShapeOfr),
                 [&](int64_t s) { return builder.getIndexAttr(s); });
  auto newExpand = builder.create<memref::ExpandShapeOp>(expandOp.getLoc(), newOutputShape, newSource,
                                                         expandOp.getReassociationIndices(), outputShapeOfr);
  replacementMap[expandOp.getResult()] = newExpand.getResult();
}

void AllocBufferShrinkPass::rebuildMemorySpaceCast(memref::MemorySpaceCastOp castOp) {
  Value newSource = replacementMap.lookup(castOp.getSource());
  if (!newSource) return;

  auto *context = &getContext();
  auto newSourceType = dyn_cast<MemRefType>(newSource.getType());
  auto oldResultType = dyn_cast<MemRefType>(castOp.getResult().getType());

  auto targetMemSpace = updateMemSpaceForShrink(oldResultType.getMemorySpace(), shrinkDims, rank, context);
  auto newResultType = MemRefType::get(newSourceType.getShape(), newSourceType.getElementType(),
                                       newSourceType.getLayout(), targetMemSpace);

  OpBuilder builder(castOp);
  auto newCast = builder.create<memref::MemorySpaceCastOp>(castOp.getLoc(), newResultType, newSource);
  replacementMap[castOp.getResult()] = newCast.getResult();
}

void AllocBufferShrinkPass::rebuildViewChain(memref::AllocOp oldAlloc, memref::AllocOp newAlloc) {
  replacementMap.clear();
  replacementMap[oldAlloc.getResult()] = newAlloc.getResult();

  for (auto *op : viewOps) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op))
      rebuildSubView(subviewOp);
    else if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(op))
      rebuildCollapseShape(collapseOp);
    else if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(op))
      rebuildExpandShape(expandOp);
    else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(op))
      rebuildMemorySpaceCast(castOp);
  }
}

void AllocBufferShrinkPass::rewriteAccessOps() {
  for (auto &info : accessInfos) {
    Value oldMemref = getAccessMemRef(info.op);
    SmallVector<Value> oldIndices = getAccessIndices(info.op);

    Value newMemref = replacementMap.lookup(oldMemref);
    if (!newMemref) newMemref = oldMemref;

    SmallVector<Value> newIndices(oldIndices);
    for (unsigned d = 0; d < rank; d++) {
      int mapped = info.dimMapping[d];
      if (shrinkDims[d] && mapped >= 0 && mapped < static_cast<int>(newIndices.size())) newIndices[mapped] = zeroIdx;
    }

    OpBuilder builder(info.op);
    Operation *newOp = nullptr;
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(info.op)) {
      auto newLoad = builder.create<affine::AffineLoadOp>(info.op->getLoc(), newMemref, newIndices);
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      newOp = newLoad;
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(info.op)) {
      auto newLoad = builder.create<memref::LoadOp>(info.op->getLoc(), newMemref, newIndices);
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      newOp = newLoad;
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(info.op)) {
      newOp =
        builder.create<affine::AffineStoreOp>(info.op->getLoc(), storeOp.getValueToStore(), newMemref, newIndices);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(info.op)) {
      newOp = builder.create<memref::StoreOp>(info.op->getLoc(), storeOp.getValueToStore(), newMemref, newIndices);
    } else {
      continue;
    }

    for (auto namedAttr : info.op->getAttrs()) {
      if (namedAttr.getName().getValue() != "map") newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }
    info.op->erase();
  }
}

void AllocBufferShrinkPass::eraseOldOps(memref::AllocOp allocOp) {
  for (auto it = viewOps.rbegin(); it != viewOps.rend(); ++it) {
    if ((*it)->use_empty()) (*it)->erase();
  }
  if (allocOp->use_empty()) allocOp->erase();
}

// --- Entry Points -----------------------------------------------------------

bool AllocBufferShrinkPass::processAlloc(memref::AllocOp allocOp) {
  auto memrefType = allocOp.getType();
  rank = memrefType.getRank();
  if (rank == 0) return false;

  // 1. Collect all access ops and view chain.
  accessInfos.clear();
  viewOps.clear();
  SmallVector<int> identityMapping(rank);
  for (unsigned i = 0; i < rank; i++) identityMapping[i] = static_cast<int>(i);
  if (!collectAccessOpsRecursive(allocOp.getResult(), rank, identityMapping, accessInfos, viewOps)) return false;
  if (accessInfos.empty()) return false;
  if (!allAccessMapsAreIdentity()) return false;

  // 2. Determine which dimensions can be shrunk.
  analyzeShrinkableDims(memrefType);
  validateSubviewOffsets();
  if (!llvm::any_of(shrinkDims, [](bool v) { return v; })) {
    return false;
  }

  // 3. Create new alloc and ensure a zero-index constant exists.
  auto newAlloc = createShrunkAlloc(allocOp, memrefType);
  if (!zeroIdx || zeroIdx.getDefiningOp()->getBlock() != newAlloc->getBlock()) {
    OpBuilder zeroBuilder(newAlloc->getBlock(), std::next(Block::iterator(newAlloc)));
    zeroIdx = zeroBuilder.create<arith::ConstantIndexOp>(newAlloc.getLoc(), 0);
  }

  // 4. Rebuild view chain, rewrite accesses, and clean up.
  rebuildViewChain(allocOp, newAlloc);
  rewriteAccessOps();
  eraseOldOps(allocOp);
  return true;
}

void AllocBufferShrinkPass::runOnOperation() {
  zeroIdx = Value();
  SmallVector<memref::AllocOp> allocOps;
  getOperation()->walk([&](memref::AllocOp op) { allocOps.push_back(op); });

  for (auto allocOp : allocOps) processAlloc(allocOp);
}

}  // namespace

std::unique_ptr<Pass> createAllocBufferShrinkPass() { return std::make_unique<AllocBufferShrinkPass>(); }
}  // namespace mlir
