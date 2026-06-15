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

#include "akg/Dialect/Linalg/Transforms/HoistTensorSlice.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DECL_HOISTTENSORSLICE
#define GEN_PASS_DEF_HOISTTENSORSLICE
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "hoist-tensor-slice"

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

struct SliceSpec {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
};

// REDUCE: helper
static bool hasReductionIterator(linalg::GenericOp gen) {
  auto iteratorTypes = gen.getIteratorTypesArray();
  return std::any_of(iteratorTypes.begin(), iteratorTypes.end(),
                     [](utils::IteratorType it) { return it == utils::IteratorType::reduction; });
}

/// A value is considered a "leaf" for the chain if it is a BlockArgument of
/// `func`'s entry block, or a result of a ConstantLike op.
static bool isFuncLeaf(Value v, func::FuncOp func) {
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    return ba.getOwner() == &func.getBody().front();
  }
  if (Operation *op = v.getDefiningOp()) {
    if (op->hasTrait<OpTrait::ConstantLike>()) {
      return true;
    }
    if (auto expandOp = dyn_cast<tensor::ExpandShapeOp>(op)) {
      return isFuncLeaf(expandOp.getSrc(), func);
    }
    if (auto collapseOp = dyn_cast<tensor::CollapseShapeOp>(op)) {
      return isFuncLeaf(collapseOp.getSrc(), func);
    }
  }
  return false;
}

/// Build the cone of linalg.generic ops from `rootOp` upward until hitting
/// leaves (func args / constants). Populates `cone`. Returns false if any of
/// the hoisting preconditions is violated.
/// Preconditions:
///  - rootOp and every op in the cone is a linalg.generic with exactly one
///    dps init. Reduction iterators are allowed .
///  - All users of rootOp's result are in `sliceSet`.
///  - All users of intermediate cone op results are themselves in the cone.
///  - Every input operand of a cone op is either a leaf, or defined by another
///    linalg.generic admitted into the cone.
static bool buildCone(Operation *rootOp, const DenseSet<Operation *> &sliceSet, func::FuncOp func,
                      DenseSet<Operation *> &cone) {
  SmallVector<Operation *> worklist;
  worklist.push_back(rootOp);
  cone.insert(rootOp);

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();

    auto gen = dyn_cast<linalg::GenericOp>(op);
    if (!gen) {
      return false;
    }

    if (gen.getNumDpsInits() != 1) {
      return false;
    }

    // Every indexing map must be a projected permutation so that
    // computeOperandSlice's iter-space reasoning is valid.
    for (AffineMap m : gen.getIndexingMapsArray()) {
      if (!m.isProjectedPermutation(/* allowZeroInResults= */ true)) {
        return false;
      }
    }

    // User constraint:
    //   - rootOp's users must all be in sliceSet.
    //   - every other cone op's users must all be already in `cone`.
    for (Value r : op->getResults()) {
      for (Operation *u : r.getUsers()) {
        if (op == rootOp) {
          if (sliceSet.count(u) == 0u) {
            return false;
          }
        } else {
          if (cone.count(u) == 0u) {
            return false;
          }
        }
      }
    }

    // Traverse input operands (ignore DPS init/outs operands).
    for (OpOperand *opnd : gen.getDpsInputOperands()) {
      Value v = opnd->get();
      if (isFuncLeaf(v, func)) {
        continue;
      }
      Operation *d = v.getDefiningOp();
      if (d == nullptr) {
        return false;
      }
      if (cone.count(d) != 0u) {
        continue;
      }
      if (isa<linalg::GenericOp>(d)) {
        cone.insert(d);
        worklist.push_back(d);
      } else {
        return false;
      }
    }
  }
  return true;
}

// REDUCE: compute the full extent of every iteration-space dim of `op`,
// using static loop ranges where possible, and falling back to tensor.dim on
// the operand that indexes that iter-dim when the range is dynamic.
static SmallVector<OpFoldResult> computeIterSpaceExtents(linalg::GenericOp op, OpBuilder &b, Location loc) {
  unsigned numIters = op.getNumLoops();
  SmallVector<OpFoldResult> extents(numIters);
  SmallVector<int64_t> staticRanges = op.getStaticLoopRanges();

  for (unsigned it = 0; it < numIters; ++it) {
    if (!ShapedType::isDynamic(staticRanges[it])) {
      extents[it] = b.getIndexAttr(staticRanges[it]);
      continue;
    }
    // Dynamic: find some operand whose indexing map contains this iter-dim,
    // and emit a tensor.dim on the corresponding operand dim.
    Value sizeVal;
    for (OpOperand &opnd : op->getOpOperands()) {
      AffineMap m = op.getMatchingIndexingMap(&opnd);
      for (unsigned d = 0; d < m.getNumResults() && !sizeVal; ++d) {
        if (auto dimE = dyn_cast<AffineDimExpr>(m.getResult(d)); dimE && dimE.getPosition() == it) {
          sizeVal = b.createOrFold<tensor::DimOp>(loc, opnd.get(), d);
        }
      }
      if (sizeVal) {
        break;
      }
    }
    assert(sizeVal && "could not find operand indexing this iter dim");
    extents[it] = sizeVal;
  }
  return extents;
}

/// Given a linalg.generic `op`, the slice on its (single) output, and one of
/// its input operands, compute the corresponding slice on that input by
/// propagating through the indexing maps.
/// REDUCE: iter-dims that do NOT appear in the output map (i.e. reduction
/// dims) are kept at their full extent, so the upstream input is sliced only
/// along parallel dims and never along reduction dims.
static SliceSpec computeOperandSlice(linalg::GenericOp op, const SliceSpec &outSlice, OpOperand *operand, OpBuilder &b,
                                     Location loc) {
  AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  AffineMap inMap = op.getMatchingIndexingMap(operand);

  unsigned numIters = op.getNumLoops();
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult one = b.getIndexAttr(1);

  // REDUCE: start with full extents for every iter-dim, then overwrite the
  // ones that appear in outMap using outSlice.
  SmallVector<OpFoldResult> iterExtents = computeIterSpaceExtents(op, b, loc);
  SmallVector<OpFoldResult> iterOff(numIters, zero);
  SmallVector<OpFoldResult> iterSize = iterExtents;
  SmallVector<OpFoldResult> iterStr(numIters, one);

  for (unsigned d = 0; d < outMap.getNumResults(); ++d) {
    AffineExpr e = outMap.getResult(d);
    if (auto dimE = dyn_cast<AffineDimExpr>(e)) {
      unsigned it = dimE.getPosition();
      iterOff[it] = outSlice.offsets[d];
      iterSize[it] = outSlice.sizes[d];
      iterStr[it] = outSlice.strides[d];
    }
  }

  SliceSpec in;
  auto shape = cast<RankedTensorType>(operand->get().getType()).getShape();
  in.offsets.reserve(inMap.getNumResults());
  in.sizes.reserve(inMap.getNumResults());
  in.strides.reserve(inMap.getNumResults());
  for (unsigned d = 0; d < inMap.getNumResults(); ++d) {
    AffineExpr e = inMap.getResult(d);
    if (auto dimE = dyn_cast<AffineDimExpr>(e)) {
      unsigned it = dimE.getPosition();
      in.offsets.push_back(iterOff[it]);
      in.sizes.push_back(iterSize[it]);
      in.strides.push_back(iterStr[it]);
    } else if (auto cstE = dyn_cast<AffineConstantExpr>(e)) {
      // Broadcast dim: keep full (typically size 1).
      in.offsets.push_back(b.getIndexAttr(cstE.getValue()));
      if (ShapedType::isDynamic(shape[d])) {
        in.sizes.push_back(b.createOrFold<tensor::DimOp>(loc, operand->get(), d));
      } else {
        in.sizes.push_back(b.getIndexAttr(shape[d]));
      }
      in.strides.push_back(one);
    } else {
      // Conservative fallback: take full dim.
      in.offsets.push_back(zero);
      if (ShapedType::isDynamic(shape[d])) {
        in.sizes.push_back(b.createOrFold<tensor::DimOp>(loc, operand->get(), d));
      } else {
        in.sizes.push_back(b.getIndexAttr(shape[d]));
      }
      in.strides.push_back(one);
    }
  }
  return in;
}

static SmallVector<int64_t> getStaticSizes(ArrayRef<OpFoldResult> sizes) {
  SmallVector<int64_t> out;
  out.reserve(sizes.size());
  for (OpFoldResult s : sizes) {
    if (auto attr = dyn_cast<Attribute>(s)) {
      out.push_back(cast<IntegerAttr>(attr).getInt());
    } else {
      out.push_back(ShapedType::kDynamic);
    }
  }
  return out;
}

// REDUCE: build the new init operand for a sliced clone of `gen`.
//   - For pure-parallel generics, init values are don't-cares: we just create
//     a fresh tensor.empty with the new shape.
//   - For generics with any reduction iterator, init values are semantically
//     meaningful (accumulator seed). We preserve them:
//       * if the original init is linalg.fill(scalar, empty), recreate it on
//         a fresh smaller empty (cheap);
//       * otherwise fall back to extract_slice on the original init tensor.
static Value buildNewInit(linalg::GenericOp gen, const SliceSpec &outSlice, RankedTensorType newOutType,
                          ArrayRef<Value> dynSizes, OpBuilder &b, Location loc) {
  bool isReduction = hasReductionIterator(gen);
  Value origInit = gen.getDpsInitOperand(0)->get();

  if (!isReduction) {
    return b.create<tensor::EmptyOp>(loc, newOutType, dynSizes);
  }

  // Reduction path: preserve init semantics.
  if (auto fillOp = origInit.getDefiningOp<linalg::FillOp>()) {
    Value newEmpty = b.create<tensor::EmptyOp>(loc, newOutType, dynSizes);
    auto newFill = b.create<linalg::FillOp>(loc, fillOp.getInputs(), ValueRange{newEmpty});
    return newFill.getResult(0);
  }
  // General fallback: slice the original init, which retains all initial
  // values. Subsequent canonicalization can usually fold this further.
  return b.create<tensor::ExtractSliceOp>(loc, origInit, outSlice.offsets, outSlice.sizes, outSlice.strides);
}

/// Recursively materialize a sliced version of `val`.
static Value materializeSliced(Value val, const SliceSpec &slice, const DenseSet<Operation *> &cone,
                               DenseMap<Value, Value> &cache, OpBuilder &b, Location loc) {
  auto cacheIt = cache.find(val);
  if (cacheIt != cache.end()) {
    return cacheIt->second;
  }

  Value result;
  Operation *def = val.getDefiningOp();
  if ((def == nullptr) || (cone.count(def) == 0u)) {
    // Leaf: extract_slice directly from the function input / constant.
    result = b.create<tensor::ExtractSliceOp>(loc, val, slice.offsets, slice.sizes, slice.strides);
  } else {
    auto gen = cast<linalg::GenericOp>(def);

    // Recursively slice each input.
    SmallVector<Value> newInputs;
    newInputs.reserve(gen.getNumDpsInputs());
    for (OpOperand *opnd : gen.getDpsInputOperands()) {
      SliceSpec inSlice = computeOperandSlice(gen, slice, opnd, b, loc);
      Value newIn = materializeSliced(opnd->get(), inSlice, cone, cache, b, loc);
      newInputs.push_back(newIn);
    }

    // Build the new smaller output type.
    auto origType = cast<RankedTensorType>(val.getType());
    SmallVector<int64_t> newShape = getStaticSizes(slice.sizes);
    auto newOutType = RankedTensorType::get(newShape, origType.getElementType());

    SmallVector<Value> dynSizes;
    for (size_t i = 0; i < newShape.size(); ++i) {
      if (ShapedType::isDynamic(newShape[i]) && isa<Value>(slice.sizes[i])) {
        dynSizes.push_back(cast<Value>(slice.sizes[i]));
      }
    }

    // REDUCE: pick the correct init (empty for parallel, seeded for reduction).
    Value newInit = buildNewInit(gen, slice, newOutType, dynSizes, b, loc);

    // Clone the generic with the new inputs / init. Indexing maps and iterator
    // types are preserved - they are abstract over iter-space dims and remain
    // valid for the reduced iteration extents (parallel dims got smaller,
    // reduction dims are unchanged).
    auto newGen = b.create<linalg::GenericOp>(loc, TypeRange{newOutType}, newInputs, ValueRange{newInit},
                                              gen.getIndexingMapsArray(), gen.getIteratorTypesArray());

    // Copy over the body.
    IRMapping irm;
    gen.getRegion().cloneInto(&newGen.getRegion(), irm);

    result = newGen.getResult(0);
  }

  cache[val] = result;
  return result;
}

struct HoistTensorSlice : public impl::HoistTensorSliceBase<HoistTensorSlice> {
 public:
  HoistTensorSlice() {}

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Step 1: collect tensor.extract_slice ops grouped by source value.
    llvm::MapVector<Value, SmallVector<tensor::ExtractSliceOp>> bySource;
    funcOp.walk([&](tensor::ExtractSliceOp es) { bySource[es.getSource()].push_back(es); });

    for (auto &kv : bySource) {
      Value source = kv.first;
      SmallVector<tensor::ExtractSliceOp> &slices = kv.second;

      DenseSet<Operation *> sliceSet;
      for (tensor::ExtractSliceOp s : slices) {
        sliceSet.insert(s.getOperation());
      }

      // Step 2 (a): `source` is only used by these slices.
      bool allUsersAreSlices = true;
      for (Operation *u : source.getUsers()) {
        if (sliceSet.count(u) == 0u) {
          allUsersAreSlices = false;
          break;
        }
      }
      if (!allUsersAreSlices) {
        continue;
      }

      // Step 2 (b): must be produced by a linalg.generic.
      Operation *defOp = source.getDefiningOp();
      if ((defOp == nullptr) || !isa<linalg::GenericOp>(defOp)) {
        continue;
      }

      // Step 2 (c): build cone up to leaves.
      DenseSet<Operation *> cone;
      if (!buildCone(defOp, sliceSet, funcOp, cone)) {
        continue;
      }

      // Step 3: rematerialize sliced chains.
      for (tensor::ExtractSliceOp es : slices) {
        OpBuilder b(es);
        SliceSpec outSlice;
        outSlice.offsets = es.getMixedOffsets();
        outSlice.sizes = es.getMixedSizes();
        outSlice.strides = es.getMixedStrides();

        DenseMap<Value, Value> cache;
        Value newVal = materializeSliced(source, outSlice, cone, cache, b, es.getLoc());
        es.getResult().replaceAllUsesWith(newVal);
        es.erase();
      }

      // Erase dead cone ops in reverse program order.
      SmallVector<Operation *> coneVec(cone.begin(), cone.end());
      llvm::sort(coneVec, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
      for (auto it = coneVec.rbegin(); it != coneVec.rend(); ++it) {
        if ((*it)->use_empty()) {
          (*it)->erase();
        }
      }
    }

    // Final cleanup: dead tensor.empty / linalg.fill / ConstantLike.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<Operation *> toErase;
      funcOp.walk([&](Operation *op) {
        if (op->use_empty() &&
            (isa<tensor::EmptyOp>(op) || isa<linalg::FillOp>(op) || op->hasTrait<OpTrait::ConstantLike>())) {
          toErase.push_back(op);
        }
      });
      for (Operation *op : toErase) {
        op->erase();
        changed = true;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHoistTensorSlicePass() {
  return std::make_unique<HoistTensorSlice>();
}

}  // namespace mlir
