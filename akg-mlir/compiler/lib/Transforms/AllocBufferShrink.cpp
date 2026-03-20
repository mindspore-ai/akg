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

// If `idx` is the induction variable of an affine.for / scf.for, return that
// loop operation; else null.
static Operation *getOwningLoopOp(Value idx) {
  auto blockArg = dyn_cast<BlockArgument>(idx);
  if (!blockArg) return nullptr;
  auto *parentOp = blockArg.getOwner()->getParentOp();
  if (auto forOp = dyn_cast<affine::AffineForOp>(parentOp)) {
    return forOp.getInductionVar() == blockArg ? parentOp : nullptr;
  }
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    return forOp.getInductionVar() == blockArg ? parentOp : nullptr;
  }
  return nullptr;
}

static SmallVector<Value> getAccessIndices(Operation *op) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    return SmallVector<Value>(loadOp.getIndices().begin(), loadOp.getIndices().end());
  } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    return SmallVector<Value>(storeOp.getIndices().begin(), storeOp.getIndices().end());
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    return SmallVector<Value>(loadOp.getIndices().begin(), loadOp.getIndices().end());
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return SmallVector<Value>(storeOp.getIndices().begin(), storeOp.getIndices().end());
  }
  return {};
}

static Value getAccessMemRef(Operation *op) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
    return loadOp.getMemRef();
  } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
    return storeOp.getMemRef();
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    return loadOp.getMemRef();
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return storeOp.getMemRef();
  }
  return {};
}

// Recursively trace through view-like ops (SubViewOp with unit strides and
// same rank, MemorySpaceCastOp) to collect:
//   - terminal affine/memref load/store users -> accessOps
//   - intermediate view ops (in definition order) -> viewOps
// Returns false if any unsupported user is encountered.
static bool collectAccessOpsRecursive(Value memref, unsigned allocRank, SmallVectorImpl<Operation *> &accessOps,
                                      SmallVectorImpl<Operation *> &viewOps) {
  for (auto *user : memref.getUsers()) {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, memref::LoadOp, memref::StoreOp>(user)) {
      accessOps.push_back(user);
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
      auto sourceRank = subviewOp.getSourceType().getRank();
      auto targetRank = subviewOp.getType().getRank();
      if (sourceRank != targetRank || targetRank != allocRank) {
        return false;
      }
      auto strides = subviewOp.getStaticStrides();
      if (!llvm::all_of(strides, [](int64_t s) { return s == 1; })) {
        return false;
      }
      viewOps.push_back(user);
      if (!collectAccessOpsRecursive(subviewOp.getResult(), allocRank, accessOps, viewOps)) {
        return false;
      }
    } else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(user)) {
      viewOps.push_back(user);
      if (!collectAccessOpsRecursive(castOp.getResult(), allocRank, accessOps, viewOps)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

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
  for (unsigned d = 0; d < rank; d++) {
    newSymShapes.push_back(shrinkDims[d] ? StringAttr::get(context, "1") : symShapeAttr[d]);
  }
  SmallVector<NamedAttribute> entries;
  for (auto entry : dictAttr) {
    if (entry.getName().getValue() == "SymShapeAttr")
      entries.push_back({entry.getName(), ArrayAttr::get(context, newSymShapes)});
    else
      entries.push_back(entry);
  }
  return DictionaryAttr::get(context, entries);
}

struct AllocBufferShrinkPass : public AllocBufferShrinkBase<AllocBufferShrinkPass> {
 public:
  void runOnOperation() override;

 private:
  // Per-alloc state, reset at the beginning of each processAlloc() call.
  unsigned rank = 0;
  SmallVector<bool> shrinkDims;
  SmallVector<Operation *> accessOps;
  SmallVector<Operation *> viewOps;
  DenseMap<Value, Value> replacementMap;
  Value zeroIdx;

  bool processAlloc(memref::AllocOp allocOp);
  bool allAccessMapsAreIdentity();
  void analyzeShrinkableDims(MemRefType memrefType);
  void validateSubviewOffsets();
  memref::AllocOp createShrunkAlloc(memref::AllocOp allocOp, MemRefType memrefType);
  void rebuildViewChain(memref::AllocOp oldAlloc, memref::AllocOp newAlloc);
  void rewriteAccessOps();
  void eraseOldOps(memref::AllocOp allocOp);
};

// Check that affine access ops use identity maps, ensuring map operands
// correspond 1-to-1 with memref dimensions. memref load/store are always OK.
bool AllocBufferShrinkPass::allAccessMapsAreIdentity() {
  for (auto *op : accessOps) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      if (!loadOp.getAffineMap().isIdentity()) {
        return false;
      }
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      if (!storeOp.getAffineMap().isIdentity()) {
        return false;
      }
    }
  }
  return true;
}

// Analyze each dimension of the alloc to determine if it can be shrunk to 1.
// A dimension is shrinkable when all accesses index it with the same loop
// induction variable and all accesses reside inside that loop.
void AllocBufferShrinkPass::analyzeShrinkableDims(MemRefType memrefType) {
  auto shape = memrefType.getShape();
  shrinkDims.assign(rank, false);

  for (unsigned d = 0; d < rank; d++) {
    if (ShapedType::isDynamic(shape[d]) || shape[d] <= 1) continue;

    Operation *commonLoop = nullptr;
    bool canShrink = true;

    for (auto *op : accessOps) {
      SmallVector<Value> indices = getAccessIndices(op);

      if (d >= indices.size()) {
        canShrink = false;
        break;
      }

      auto *loopOp = getOwningLoopOp(indices[d]);
      if (!loopOp) {
        canShrink = false;
        break;
      }

      if (!commonLoop) {
        commonLoop = loopOp;
      } else if (commonLoop != loopOp) {
        canShrink = false;
        break;
      }
    }

    if (canShrink && commonLoop) {
      for (auto *op : accessOps) {
        if (!commonLoop->isAncestor(op)) {
          canShrink = false;
          break;
        }
      }
    } else {
      canShrink = false;
    }

    shrinkDims[d] = canShrink;
  }
}

// Revoke shrinkability for dimensions where any SubViewOp in the view chain
// has a non-zero static offset.
void AllocBufferShrinkPass::validateSubviewOffsets() {
  for (auto *op : viewOps) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      auto offsets = subviewOp.getStaticOffsets();
      for (unsigned d = 0; d < rank; d++) {
        if (shrinkDims[d] && offsets[d] != 0) {
          shrinkDims[d] = false;
        }
      }
    }
  }
}

// Create the new alloc with shrunk dimensions and updated SymShapeAttr.
memref::AllocOp AllocBufferShrinkPass::createShrunkAlloc(memref::AllocOp allocOp, MemRefType memrefType) {
  auto *context = &getContext();
  auto shape = memrefType.getShape();

  SmallVector<int64_t> newShape;
  for (unsigned d = 0; d < rank; d++) {
    newShape.push_back(shrinkDims[d] ? 1 : shape[d]);
  }

  auto newMemorySpace = updateMemSpaceForShrink(memrefType.getMemorySpace(), shrinkDims, rank, context);
  auto newType = MemRefType::get(newShape, memrefType.getElementType(), memrefType.getLayout(), newMemorySpace);

  OpBuilder builder(allocOp);
  auto newAlloc = builder.create<memref::AllocOp>(allocOp.getLoc(), newType);
  if (auto alignAttr = allocOp.getAlignmentAttr()) {
    newAlloc->setAttr("alignment", alignAttr);
  }
  return newAlloc;
}

// Rebuild the view chain (SubViewOp / MemorySpaceCastOp) with updated types
// for shrunk dimensions.
void AllocBufferShrinkPass::rebuildViewChain(memref::AllocOp oldAlloc, memref::AllocOp newAlloc) {
  auto *context = &getContext();
  replacementMap.clear();
  replacementMap[oldAlloc.getResult()] = newAlloc.getResult();

  for (auto *op : viewOps) {
    if (auto subviewOp = dyn_cast<memref::SubViewOp>(op)) {
      Value newSource = replacementMap.lookup(subviewOp.getSource());
      if (!newSource) continue;

      auto newSourceType = dyn_cast<MemRefType>(newSource.getType());
      auto oldResultType = subviewOp.getType();

      SmallVector<OpFoldResult> newOffsets = subviewOp.getMixedOffsets();
      SmallVector<OpFoldResult> newSizes = subviewOp.getMixedSizes();
      SmallVector<OpFoldResult> newStrides = subviewOp.getMixedStrides();
      for (unsigned d = 0; d < rank; d++) {
        if (shrinkDims[d]) {
          newSizes[d] = OpBuilder(op).getI64IntegerAttr(1);
        }
      }

      SmallVector<int64_t> newResultShape;
      for (unsigned d = 0; d < rank; d++) {
        newResultShape.push_back(shrinkDims[d] ? 1 : oldResultType.getShape()[d]);
      }

      auto newResultType = MemRefType::get(newResultShape, oldResultType.getElementType(), oldResultType.getLayout(),
                                           newSourceType.getMemorySpace());

      OpBuilder builder(subviewOp);
      auto newSubview = builder.create<memref::SubViewOp>(subviewOp.getLoc(), newResultType, newSource, newOffsets,
                                                          newSizes, newStrides);
      replacementMap[subviewOp.getResult()] = newSubview.getResult();

    } else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(op)) {
      Value newSource = replacementMap.lookup(castOp.getSource());
      if (!newSource) continue;

      auto newSourceType = dyn_cast<MemRefType>(newSource.getType());
      auto oldResultType = dyn_cast<MemRefType>(castOp.getResult().getType());

      auto targetMemSpace = updateMemSpaceForShrink(oldResultType.getMemorySpace(), shrinkDims, rank, context);
      auto newResultType = MemRefType::get(newSourceType.getShape(), newSourceType.getElementType(),
                                           newSourceType.getLayout(), targetMemSpace);

      OpBuilder builder(castOp);
      auto newCast = builder.create<memref::MemorySpaceCastOp>(castOp.getLoc(), newResultType, newSource);
      replacementMap[castOp.getResult()] = newCast.getResult();
    }
  }
}

// Rewrite access ops: replace the memref operand and collapse every shrunk
// dimension to a constant-0 index, preserving the original op type.
void AllocBufferShrinkPass::rewriteAccessOps() {
  for (auto *op : accessOps) {
    Value oldMemref = getAccessMemRef(op);
    SmallVector<Value> oldIndices = getAccessIndices(op);

    Value newMemref = replacementMap.lookup(oldMemref);
    if (!newMemref) newMemref = oldMemref;

    OpBuilder builder(op);
    SmallVector<Value> newIndices;
    for (unsigned d = 0; d < rank; d++) {
      if (shrinkDims[d]) {
        newIndices.push_back(zeroIdx);
      } else {
        newIndices.push_back(oldIndices[d]);
      }
    }

    Operation *newOp = nullptr;
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      auto newLoad = builder.create<affine::AffineLoadOp>(op->getLoc(), newMemref, newIndices);
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      newOp = newLoad;
    } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      auto newLoad = builder.create<memref::LoadOp>(op->getLoc(), newMemref, newIndices);
      loadOp.getResult().replaceAllUsesWith(newLoad.getResult());
      newOp = newLoad;
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
      newOp = builder.create<affine::AffineStoreOp>(op->getLoc(), storeOp.getValueToStore(), newMemref, newIndices);
    } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
      newOp = builder.create<memref::StoreOp>(op->getLoc(), storeOp.getValueToStore(), newMemref, newIndices);
    } else {
      continue;
    }

    for (auto namedAttr : op->getAttrs()) {
      if (namedAttr.getName().getValue() == "map") {
        continue;
      }
      newOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    op->erase();
  }
}

// Clean up: erase old view ops (in reverse order) and the old alloc.
void AllocBufferShrinkPass::eraseOldOps(memref::AllocOp allocOp) {
  for (auto it = viewOps.rbegin(); it != viewOps.rend(); ++it) {
    if ((*it)->use_empty()) (*it)->erase();
  }
  if (allocOp->use_empty()) allocOp->erase();
}

bool AllocBufferShrinkPass::processAlloc(memref::AllocOp allocOp) {
  auto memrefType = allocOp.getType();
  rank = memrefType.getRank();
  if (rank == 0) return false;

  accessOps.clear();
  viewOps.clear();
  if (!collectAccessOpsRecursive(allocOp.getResult(), rank, accessOps, viewOps)) return false;
  if (accessOps.empty()) return false;

  if (!allAccessMapsAreIdentity()) return false;

  analyzeShrinkableDims(memrefType);
  validateSubviewOffsets();
  if (!llvm::any_of(shrinkDims, [](bool v) { return v; })) {
    return false;
  }

  auto newAlloc = createShrunkAlloc(allocOp, memrefType);

  if (!zeroIdx || zeroIdx.getDefiningOp()->getBlock() != newAlloc->getBlock()) {
    OpBuilder zeroBuilder(newAlloc->getBlock(), std::next(Block::iterator(newAlloc)));
    zeroIdx = zeroBuilder.create<arith::ConstantIndexOp>(newAlloc.getLoc(), 0);
  }

  rebuildViewChain(allocOp, newAlloc);
  rewriteAccessOps();
  eraseOldOps(allocOp);
  return true;
}

void AllocBufferShrinkPass::runOnOperation() {
  zeroIdx = Value();
  SmallVector<memref::AllocOp> allocOps;
  getOperation()->walk([&](memref::AllocOp op) { allocOps.push_back(op); });

  for (auto allocOp : allocOps) {
    processAlloc(allocOp);
  }
}

}  // namespace

std::unique_ptr<Pass> createAllocBufferShrinkPass() { return std::make_unique<AllocBufferShrinkPass>(); }
}  // namespace mlir
