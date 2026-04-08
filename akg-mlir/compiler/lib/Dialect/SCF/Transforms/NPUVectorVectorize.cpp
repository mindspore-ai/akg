/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// NPUVectorVectorize.cpp
//
// SCF loop vectorization pass using NPUVector Dialect.
// Supports N-D vectorization, dynamic shapes, and automatic transpose detection.
//
// Architecture:
//   1. Analyze loop attributes and eligibility
//   2. Determine vectorization strategy
//      ├── 1-D: single loop with vector/reduction_x/reduction_y/broadcast attribute
//      └── N-D: nested loops each with vector=N, collected via
//               collectVectorizationStrategy into a multi-dim VectorizationContext
//               (broadcast/reduction_y inner loops are not collected as vector dims)
//   3. Create and populate vectorized loop
//      ├── vectorizeLoad → npuvector.transfer_read (+ npuvector.transpose if needed)
//      ├── vectorizeStore → npuvector.transfer_write (+ npuvector.transpose if needed)
//      ├── vectorizeArithOp → type conversion and arithmetic operations
//      └── vectorizeBroadcastScalar → npuvector.broadcast
//   4. Finalization
//      ├── Elementwise: inline or loop-based transformation
//      └── Reduction: vector reduction + tail processing + init value merging
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <iterator>
#include <numeric>

#include "akg/Dialect/SCF/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "npuvector-vectorize"

namespace mlir {
namespace scf {
#define GEN_PASS_DECL_NPUVECTORVECTORIZEPASS
#define GEN_PASS_DEF_NPUVECTORVECTORIZEPASS
#include "akg/Dialect/SCF/Passes.h.inc"
}  // namespace scf
}  // namespace mlir

using namespace mlir;  // NOLINT(build/namespaces)

namespace {

enum class VectorizationMode {
  None,
  Elementwise,
  ReductionX,
  ReductionY,
  /// Inner loop: IV stays scalar; mem ops vectorize along parent axis (like nested
  /// reduction_y mapping) but no reduction iter_args — finalize like elementwise.
  Broadcast,
};

struct VectorizationContext {
  OpBuilder &builder;
  IRMapping valueMapping;

  SmallVector<int64_t> vectorSizes;
  SmallVector<Value> vectorSizeValues;   // nullptr for static dims
  SmallVector<Value> maxStepValues;

  DenseMap<Operation *, unsigned> loopToVectorDim;

  VectorizationMode mode;

  scf::ForOp scalarLoop;
  scf::ForOp vecLoop;

  std::optional<arith::AtomicRMWKind> reductionKind;
  Value origInit;

  Value vectorizationAxis;

  // Tracks the logical dimension order of each vectorized Value.
  // E.g. [1, 0, 2] means vector dim 0 holds data for logical dim 1, etc.
  // Used by vectorizeStore to compute the correct transpose permutation.
  DenseMap<Value, SmallVector<int>> valueDimOrder;

  VectorizationContext(OpBuilder &b, SmallVector<int64_t> vecSizes,
                      SmallVector<Value> vecSizeVals, SmallVector<Value> maxVals,
                      VectorizationMode m, scf::ForOp loop, Value vecAxis = nullptr)
      : builder(b), vectorSizes(std::move(vecSizes)),
        vectorSizeValues(std::move(vecSizeVals)), maxStepValues(std::move(maxVals)),
        mode(m), scalarLoop(loop), vecLoop(nullptr),
        reductionKind(std::nullopt), origInit(nullptr), vectorizationAxis(vecAxis) {}

  VectorizationContext(OpBuilder &b, int64_t actualStepVal, Value vfVal, Value maxVal,
                      VectorizationMode m, scf::ForOp loop, Value vecAxis = nullptr)
      : builder(b), vectorSizes({actualStepVal}),
        vectorSizeValues({vfVal}), maxStepValues({maxVal}),
        mode(m), scalarLoop(loop), vecLoop(nullptr),
        reductionKind(std::nullopt), origInit(nullptr), vectorizationAxis(vecAxis) {}

  bool isDynamic() const {
    return llvm::any_of(vectorSizeValues, [](Value v) { return v != nullptr; });
  }

  int64_t getRank() const { return vectorSizes.size(); }

  int64_t getActualStep() const { return vectorSizes.back(); }
  Value getVectorSizeValue() const { return vectorSizeValues.back(); }
  Value getMaxStepValue() const { return maxStepValues.back(); }

  Value getVectorizationAxis() {
    if (vectorizationAxis &&
        (mode == VectorizationMode::ReductionY ||
         mode == VectorizationMode::Broadcast)) {
      return vectorizationAxis;
    }
    return scalarLoop.getInductionVar();
  }

  npuvector::NPUVectorType getVectorType(Type elemType) const {
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < vectorSizes.size(); ++i) {
      if (vectorSizeValues[i])
        shape.push_back(ShapedType::kDynamic);
      else
        shape.push_back(vectorSizes[i]);
    }
    return npuvector::NPUVectorType::get(shape, elemType);
  }

  int getVectorDimForIV(Value iv) const {
    for (auto &[op, dim] : loopToVectorDim) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (forOp.getInductionVar() == iv) return static_cast<int>(dim);
      }
    }
    return -1;
  }
};

static Value vectorizeBroadcastScalar(Value scalarVal, VectorizationContext &ctx,
                                      npuvector::NPUVectorType targetType = {});
static void ensureReductionYield(VectorizationContext &ctx);
static LogicalResult vectorizeLoop(scf::ForOp scalarLoop, int64_t actualStep, Value vfValue,
                                   Value maxStepValue, VectorizationMode mode,
                                   OpBuilder &builder, Value vecAxis, Value outerIVMapping = Value());
static LogicalResult vectorizeLoopMultiDim(scf::ForOp scalarLoop,
                                           VectorizationContext &parentCtx);

static bool hasVectorizationAttr(Operation *op) {
  return op->hasAttr(kVectorAttr) ||
         op->hasAttr(kBroadcastLoopAttr) ||
         op->hasAttr(kReductionXLoopAttr) ||
         op->hasAttr(kReductionYLoopAttr) ||
         op->hasAttr(kReductionAllLoopAttr);
}

static bool shouldCloneScalarOp(Operation &op, VectorizationContext &ctx) {
  if (isa<affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp>(&op)) {
    return true;
  }
  if (op.getNumResults() > 0 && op.getResult(0).getType().isIndex()) {
    return true;
  }
  return op.hasAttr(kSkipVectorizeAttr);
}

static int64_t computeStaticVectorSize(scf::ForOp loop, int64_t maxStep) {
  auto ubConst = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lbConst = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();

  if (ubConst && lbConst) {
    int64_t tripCount = ubConst.value() - lbConst.value();
    return std::min(tripCount, maxStep);
  }

  return maxStep;
}

static Value computeDynamicVectorSize(scf::ForOp loop, Value maxStepValue,
                                     OpBuilder &builder, Location loc) {
  Value upperBound = loop.getUpperBound();
  Value lowerBound = loop.getLowerBound();

  Value tripCount = builder.create<arith::SubIOp>(loc, upperBound, lowerBound);

  Value vectorSize = builder.create<arith::MinSIOp>(loc, tripCount, maxStepValue);

  return vectorSize;
}

/// Overload: use mapped bounds instead of loop. Required when bounds are from
/// the original block that will be erased; using mapped values avoids dangling refs.
static Value computeDynamicVectorSize(Value upperBound, Value lowerBound,
                                     Value maxStepValue, OpBuilder &builder,
                                     Location loc) {
  Value tripCount = builder.create<arith::SubIOp>(loc, upperBound, lowerBound);
  Value vectorSize = builder.create<arith::MinSIOp>(loc, tripCount, maxStepValue);
  return vectorSize;
}

static VectorizationMode getVectorizationMode(scf::ForOp loop, int64_t &maxStepFromAttr) {
  if (loop->hasAttr(kReductionXLoopAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kReductionXLoopAttr);
    if (!attr) return VectorizationMode::None;
    maxStepFromAttr = attr.getInt();
    return VectorizationMode::ReductionX;
  }

  if (loop->hasAttr(kReductionYLoopAttr)) {
    if (auto attr = loop->getAttrOfType<IntegerAttr>(kReductionYLoopAttr))
      maxStepFromAttr = attr.getInt();
    else
      maxStepFromAttr = kVectorSize;
    return VectorizationMode::ReductionY;
  }

  if (loop->hasAttr(kReductionAllLoopAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kReductionAllLoopAttr);
    if (!attr) return VectorizationMode::None;
    maxStepFromAttr = attr.getInt();
    return VectorizationMode::ReductionX;
  }

  if (loop->hasAttr(kBroadcastLoopAttr)) {
    if (auto attr = loop->getAttrOfType<IntegerAttr>(kBroadcastLoopAttr))
      maxStepFromAttr = attr.getInt();
    else
      maxStepFromAttr = kVectorSize;
    return VectorizationMode::Broadcast;
  }

  if (loop->hasAttr(kVectorAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kVectorAttr);
    if (!attr) return VectorizationMode::None;
    maxStepFromAttr = attr.getInt();
    return VectorizationMode::Elementwise;
  }

  return VectorizationMode::None;
}

static std::pair<bool, bool> checkLoopEligibility(scf::ForOp loop) {
  auto stepConst = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!stepConst || stepConst.value() != 1) {
    return {true, false};
  }

  auto ubConst = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lbConst = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();

  bool isDynamic = !(ubConst && lbConst);

  return {false, isDynamic};
}

static std::optional<arith::AtomicRMWKind> detectReductionKind(scf::ForOp loop) {
  if (loop.getInitArgs().empty()) {
    return std::nullopt;
  }

  auto yieldOp = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() == 0) {
    return std::nullopt;
  }

  Value yieldValue = yieldOp.getOperand(0);
  Operation *defOp = yieldValue.getDefiningOp();

  if (!defOp) {
    return std::nullopt;
  }

  Value iterArg = loop.getRegionIterArgs()[0];

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    if (addOp.getLhs() == iterArg || addOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::addf;
    }
  }

  if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
    if (mulOp.getLhs() == iterArg || mulOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::mulf;
    }
  }

  if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
    if (maxOp.getLhs() == iterArg || maxOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::maximumf;
    }
  }

  if (auto minOp = dyn_cast<arith::MinimumFOp>(defOp)) {
    if (minOp.getLhs() == iterArg || minOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::minimumf;
    }
  }

  return std::nullopt;
}

static Value createNeutralValue(
    arith::AtomicRMWKind kind,
    Type elemType,
    VectorizationContext &ctx,
    Location loc) {

  Attribute neutralAttr = arith::getIdentityValueAttr(kind, elemType, ctx.builder, loc);

  npuvector::NPUVectorType vecType = ctx.getVectorType(elemType);

  if (ctx.isDynamic()) {
    Value neutralScalar = ctx.builder.create<arith::ConstantOp>(
        loc, mlir::cast<TypedAttr>(neutralAttr));

    SmallVector<Value> dynamicSizes;
    SmallVector<Value> maxSizes;
    for (unsigned i = 0; i < ctx.vectorSizes.size(); ++i) {
      maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.maxStepValues[i]));
      if (ctx.vectorSizeValues[i])
        dynamicSizes.push_back(
            ctx.valueMapping.lookupOrDefault(ctx.vectorSizeValues[i]));
      else
        dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(
            loc, ctx.vectorSizes[i]));
    }
    Value neutralVec = ctx.builder.create<npuvector::BroadcastOp>(
        loc, vecType, neutralScalar, ValueRange(dynamicSizes), ValueRange(maxSizes));

    return neutralVec;

  } else {
    auto vecAttr = DenseElementsAttr::get(vecType, neutralAttr);
    Value neutralVec = ctx.builder.create<arith::ConstantOp>(loc, vecAttr);

    return neutralVec;
  }
}

static bool isNeutralElement(arith::AtomicRMWKind kind, Value value,
                               OpBuilder &builder) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  Attribute neutralAttr = arith::getIdentityValueAttr(
      kind, value.getType(), builder, value.getLoc());

  bool isNeutral = (constOp.getValue() == neutralAttr);

  return isNeutral;
}

static vector::CombiningKind convertToCombiningKind(arith::AtomicRMWKind kind) {
  switch (kind) {
    case arith::AtomicRMWKind::addf:
    case arith::AtomicRMWKind::addi:
      return vector::CombiningKind::ADD;

    case arith::AtomicRMWKind::mulf:
    case arith::AtomicRMWKind::muli:
      return vector::CombiningKind::MUL;

    case arith::AtomicRMWKind::maximumf:
      return vector::CombiningKind::MAXIMUMF;

    case arith::AtomicRMWKind::minimumf:
      return vector::CombiningKind::MINIMUMF;

    case arith::AtomicRMWKind::maxs:
      return vector::CombiningKind::MAXSI;

    case arith::AtomicRMWKind::mins:
      return vector::CombiningKind::MINSI;

    case arith::AtomicRMWKind::maxu:
      return vector::CombiningKind::MAXUI;

    case arith::AtomicRMWKind::minu:
      return vector::CombiningKind::MINUI;

    default:
      llvm_unreachable("Unsupported reduction kind");
  }
}

static scf::ForOp createEmptyVectorizedLoop(VectorizationContext &ctx) {
  Location loc = ctx.scalarLoop.getLoc();

  Value newStepValue;
  if (ctx.mode == VectorizationMode::ReductionY ||
      ctx.mode == VectorizationMode::Broadcast) {
    newStepValue = ctx.scalarLoop.getStep();
  } else {
    unsigned dim = ctx.loopToVectorDim.lookup(ctx.scalarLoop);
    if (ctx.vectorSizeValues[dim]) {
      newStepValue = ctx.vectorSizeValues[dim];
    } else {
      newStepValue =
          ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.vectorSizes[dim]);
    }
  }

  Value neutralVec;
  if (ctx.mode == VectorizationMode::ReductionX ||
      ctx.mode == VectorizationMode::ReductionY) {
    if (!ctx.scalarLoop.getInitArgs().empty()) {
      auto kind = detectReductionKind(ctx.scalarLoop);
      if (!kind) return nullptr;

      Type elemType = ctx.scalarLoop.getRegionIterArgs()[0].getType();
      neutralVec = createNeutralValue(*kind, elemType, ctx, loc);
      if (!neutralVec) return nullptr;

      ctx.reductionKind = kind;
      ctx.origInit = ctx.scalarLoop.getInitArgs()[0];
    }
  }

  Value upperBound = ctx.scalarLoop.getUpperBound();
  Value lowerBound = ctx.scalarLoop.getLowerBound();

  if (ctx.mode == VectorizationMode::ReductionX) {
    if (!ctx.isDynamic()) {
      auto ubConst = upperBound.getDefiningOp<arith::ConstantIndexOp>();
      auto lbConst = lowerBound.getDefiningOp<arith::ConstantIndexOp>();
      if (ubConst && lbConst) {
        int64_t lbVal = lbConst.value();
        int64_t ubVal = ubConst.value();
        int64_t tripCount = ubVal - lbVal;
        int64_t alignedTripCount = (tripCount / ctx.getActualStep()) * ctx.getActualStep();
        int64_t alignedUb = lbVal + alignedTripCount;
        if (alignedUb < ubVal) {
          upperBound = ctx.builder.create<arith::ConstantIndexOp>(loc, alignedUb);
        }
      }
    } else {
      Value tripCount = ctx.builder.create<arith::SubIOp>(loc, upperBound, lowerBound);
      Value numIterations = ctx.builder.create<arith::DivSIOp>(loc, tripCount, ctx.getVectorSizeValue());
      Value alignedTripCount = ctx.builder.create<arith::MulIOp>(loc, numIterations, ctx.getVectorSizeValue());
      upperBound = ctx.builder.create<arith::AddIOp>(loc, alignedTripCount, lowerBound);
    }
  }

  scf::ForOp vecLoop;
  if (neutralVec) {
    vecLoop = ctx.builder.create<scf::ForOp>(
        loc, lowerBound, upperBound, newStepValue,
        ValueRange{neutralVec},
        [](OpBuilder &, Location, Value, ValueRange) {});
  } else {
    vecLoop = ctx.builder.create<scf::ForOp>(
        loc, lowerBound, upperBound, newStepValue,
        std::nullopt,
        [](OpBuilder &, Location, Value, ValueRange) {});
  }

  return vecLoop;
}

static void collectLoadIndexVectorDims(memref::LoadOp loadOp,
                                       const VectorizationContext &ctx,
                                       SmallVectorImpl<int> &indexToDim,
                                       SmallVector<int> &activeInAppearOrder) {
  indexToDim.clear();
  activeInAppearOrder.clear();
  for (Value idx : loadOp.getIndices()) {
    int dim = ctx.getVectorDimForIV(idx);
    indexToDim.push_back(dim);
    if (dim >= 0) activeInAppearOrder.push_back(dim);
  }
}

static bool loadIndicesAreLoopInvariant(memref::LoadOp loadOp,
                                        VectorizationContext &ctx) {
  Value vecAxis = ctx.getVectorizationAxis();
  Value mappedVecAxis = ctx.valueMapping.lookupOrDefault(vecAxis);
  for (Value idx : loadOp.getIndices()) {
    if (ctx.getVectorDimForIV(idx) >= 0) return false;
    if (idx == vecAxis || idx == mappedVecAxis) return false;
    if (idx.getParentBlock() == ctx.scalarLoop.getBody()) return false;
  }
  return true;
}

static Value vectorizeLoad(memref::LoadOp loadOp, VectorizationContext &ctx) {
  MemRefType memRefType = loadOp.getMemRefType();
  Type elemType = memRefType.getElementType();
  Location loc = loadOp.getLoc();

  Value memRef = loadOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);

  if (loadIndicesAreLoopInvariant(loadOp, ctx)) {
    SmallVector<Value> indices;
    for (Value idx : loadOp.getIndices()) {
      Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
      indices.push_back(newIdx);
    }

    Value scalarLoad = ctx.builder.create<memref::LoadOp>(
        loc, mappedMemRef, indices);

    return vectorizeBroadcastScalar(scalarLoad, ctx);
  }

  SmallVector<int> indexToDim;
  SmallVector<int> activeInAppearOrder;
  collectLoadIndexVectorDims(loadOp, ctx, indexToDim, activeInAppearOrder);

  const unsigned subRank = activeInAppearOrder.size();
  if (subRank == 0) {
    SmallVector<Value> indices;
    indices.reserve(loadOp.getIndices().size());
    std::transform(loadOp.getIndices().begin(), loadOp.getIndices().end(),
                   std::back_inserter(indices), [&ctx](Value idx) {
                     return ctx.valueMapping.lookupOrDefault(idx);
                   });
    Value scalarLoad =
        ctx.builder.create<memref::LoadOp>(loc, mappedMemRef, indices);
    return vectorizeBroadcastScalar(scalarLoad, ctx);
  }

  SmallVector<int> sortedDims(activeInAppearOrder);
  llvm::sort(sortedDims);
  const bool analyzedTranspose =
      (subRank > 1) && (activeInAppearOrder != sortedDims);

  SmallVector<int64_t> readShapeWithDyn(subRank);
  for (unsigned k = 0; k < subRank; ++k) {
    int gid = activeInAppearOrder[k];
    Value dynVal = ctx.vectorSizeValues[gid];
    readShapeWithDyn[k] =
        dynVal ? ShapedType::kDynamic : ctx.vectorSizes[gid];
  }
  npuvector::NPUVectorType readVecType =
      npuvector::NPUVectorType::get(readShapeWithDyn, elemType);

  SmallVector<Value> indices;
  for (Value idx : loadOp.getIndices()) {
    Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
    indices.push_back(newIdx);
  }

  Value padding = ctx.builder.create<arith::ConstantOp>(
      loc,
      ctx.builder.getZeroAttr(elemType));

  // One extent + one max per vectorized axis (rank == subRank), including constants for
  // static tile axes so e.g. 512x?x64 passes three pairs, not only the `?` slot.
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  dynamicSizes.reserve(subRank);
  maxSizes.reserve(subRank);
  for (unsigned k = 0; k < subRank; ++k) {
    int gid = activeInAppearOrder[k];
    maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.maxStepValues[gid]));
    if (ctx.vectorSizeValues[gid])
      dynamicSizes.push_back(
          ctx.valueMapping.lookupOrDefault(ctx.vectorSizeValues[gid]));
    else
      dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(
          loc, ctx.vectorSizes[gid]));
  }

  auto transferRead = ctx.builder.create<npuvector::TransferReadOp>(
      loc,
      readVecType,
      mappedMemRef,
      ValueRange(indices),
      padding,
      Value(),
      ValueRange(dynamicSizes),
      ValueRange(maxSizes));

  Value result = transferRead.getResult();
  if (analyzedTranspose) {
    SmallVector<int64_t> transposePerm(subRank);
    for (unsigned i = 0; i < subRank; ++i) {
      int targetGlobal = sortedDims[i];
      unsigned j = 0;
      for (; j < subRank; ++j) {
        if (activeInAppearOrder[j] == targetGlobal) break;
      }
      transposePerm[i] = static_cast<int64_t>(j);
    }
    result = ctx.builder.create<npuvector::TransposeOp>(loc, result,
                                                          transposePerm);
  }
  ctx.valueDimOrder[result] = sortedDims;

  return result;
}

static void vectorizeStore(memref::StoreOp storeOp, VectorizationContext &ctx) {
  Location loc = storeOp.getLoc();

  Value storeValue = storeOp.getValue();
  Value vectorValue = ctx.valueMapping.lookupOrNull(storeValue);

  if (!vectorValue) {
    vectorValue = vectorizeBroadcastScalar(storeValue, ctx);
    if (!vectorValue) {
      return;
    }
    ctx.valueMapping.map(storeValue, vectorValue);
  }

  SmallVector<Value> indices;
  indices.reserve(storeOp.getIndices().size());
  std::transform(storeOp.getIndices().begin(), storeOp.getIndices().end(),
                std::back_inserter(indices),
                [&ctx](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });

  auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!npuVecType) {
    llvm_unreachable("vectorizeStore: vector value must be NPUVectorType");
  }

  SmallVector<int> storeDimOrder;
  for (Value idx : storeOp.getIndices()) {
    int dim = ctx.getVectorDimForIV(idx);
    if (dim >= 0) storeDimOrder.push_back(dim);
  }

  if (storeDimOrder.size() > 1) {
    SmallVector<int> valueDimOrd;
    if (ctx.valueDimOrder.count(vectorValue)) {
      valueDimOrd = ctx.valueDimOrder[vectorValue];
    } else {
      unsigned r = npuVecType.getShape().size();
      valueDimOrd.resize(r);
      std::iota(valueDimOrd.begin(), valueDimOrd.end(), 0);
    }

    if (storeDimOrder != valueDimOrd) {
      SmallVector<int64_t> perm(storeDimOrder.size());
      for (unsigned i = 0; i < storeDimOrder.size(); ++i) {
        for (unsigned j = 0; j < valueDimOrd.size(); ++j) {
          if (valueDimOrd[j] == storeDimOrder[i]) {
            perm[i] = static_cast<int64_t>(j);
            break;
          }
        }
      }
      vectorValue =
          ctx.builder.create<npuvector::TransposeOp>(loc, vectorValue, perm);
    }
  }

  Value memRef = storeOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);

  ctx.builder.create<npuvector::TransferWriteOp>(
      loc,
      Type(),
      vectorValue,
      mappedMemRef,
      ValueRange(indices),
      Value());
}

static Value vectorizeArithOp(Operation *op, VectorizationContext &ctx) {
  Location loc = op->getLoc();

  SmallVector<Value, 4> vecOperands;
  SmallVector<unsigned> scalarIndices;
  for (auto [i, operand] : llvm::enumerate(op->getOperands())) {
    Value vecOperand = ctx.valueMapping.lookupOrNull(operand);

    if (!vecOperand || !mlir::isa<npuvector::NPUVectorType>(vecOperand.getType())) {
      Value valueToBroadcast = vecOperand ? vecOperand : operand;
      vecOperands.push_back(valueToBroadcast);
      scalarIndices.push_back(i);
    } else {
      vecOperands.push_back(vecOperand);
    }
  }

  npuvector::NPUVectorType refVecType;
  for (Value vo : vecOperands) {
    if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vo.getType())) {
      refVecType = nvt;
      break;
    }
  }

  for (unsigned idx : scalarIndices) {
    Value broadcasted = vectorizeBroadcastScalar(vecOperands[idx], ctx, refVecType);
    if (!broadcasted) return nullptr;
    vecOperands[idx] = broadcasted;
  }

  SmallVector<Type, 4> vecResultTypes;
  for (Value result : op->getResults()) {
    Type scalarType = result.getType();
    npuvector::NPUVectorType vecType = refVecType
        ? npuvector::NPUVectorType::get(refVecType.getShape(), scalarType)
        : ctx.getVectorType(scalarType);
    vecResultTypes.push_back(vecType);
  }

  bool needsRename = isa<arith::ExtFOp, arith::TruncFOp,
                         arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                         arith::SIToFPOp, arith::UIToFPOp,
                         arith::FPToSIOp, arith::FPToUIOp,
                         arith::BitcastOp, arith::IndexCastOp, arith::IndexCastUIOp,
                         arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op);

  Operation *vecOp;
  if (needsRename) {
    StringRef arithOpName = op->getName().getStringRef();
    std::string npuvectorOpName = "npuvector." + arithOpName.split('.').second.str();
    OperationName npuvectorName(npuvectorOpName, ctx.builder.getContext());

    OperationState state(loc, npuvectorName);
    state.addOperands(vecOperands);
    state.addTypes(vecResultTypes);
    state.addAttributes(op->getAttrs());

    vecOp = ctx.builder.create(state);
  } else {
    vecOp = ctx.builder.create(
        loc,
        op->getName().getIdentifier(),
        vecOperands,
        vecResultTypes,
        op->getAttrs());
  }

  if (!vecOp || vecOp->getNumResults() == 0) {
    return nullptr;
  }

  Value result = vecOp->getResult(0);
  for (Value vo : vecOperands) {
    if (isa<npuvector::NPUVectorType>(vo.getType()) && ctx.valueDimOrder.count(vo)) {
      ctx.valueDimOrder[result] = ctx.valueDimOrder[vo];
      break;
    }
  }
  return result;
}

static Value vectorizeBroadcastScalar(Value scalarVal, VectorizationContext &ctx,
                                      npuvector::NPUVectorType targetType) {
  if (mlir::isa<npuvector::NPUVectorType>(scalarVal.getType())) {
    return scalarVal;
  }

  Type scalarType = scalarVal.getType();
  Location loc = scalarVal.getLoc();

  npuvector::NPUVectorType vecType = targetType
      ? npuvector::NPUVectorType::get(targetType.getShape(), scalarType)
      : ctx.getVectorType(scalarType);

  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  for (unsigned i = 0; i < ctx.vectorSizes.size(); ++i) {
    maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.maxStepValues[i]));
    if (ctx.vectorSizeValues[i])
      dynamicSizes.push_back(
          ctx.valueMapping.lookupOrDefault(ctx.vectorSizeValues[i]));
    else
      dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(
          loc, ctx.vectorSizes[i]));
  }
  auto broadcast = ctx.builder.create<npuvector::BroadcastOp>(
      loc,
      vecType,
      scalarVal,
      ValueRange(dynamicSizes),
      ValueRange(maxSizes));

  return broadcast.getResult();
}

static void cloneScalarOp(Operation &op, VectorizationContext &ctx) {
  OpBuilder::InsertionGuard guard(ctx.builder);

  IRMapping mapper;
  for (const auto &kv : ctx.valueMapping.getValueMap())
    mapper.map(kv.first, kv.second);

  Operation *clonedOp = ctx.builder.clone(op, mapper);

  clonedOp->removeAttr(kSkipVectorizeAttr);

  for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
    Value mappedValue = ctx.valueMapping.lookupOrDefault(operand);
    clonedOp->setOperand(idx, mappedValue);
  }

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    ctx.valueMapping.map(op.getResult(i), clonedOp->getResult(i));
}

static LogicalResult vectorizeRegion(Region &region, VectorizationContext &ctx);

static Value vectorizeIf(scf::IfOp ifOp, VectorizationContext &ctx) {
  Location loc = ifOp.getLoc();
  Value condition = ifOp.getCondition();
  Value vecCondition = ctx.valueMapping.lookupOrNull(condition);
  if (!vecCondition) {
    vecCondition = condition;
  }

  bool hasElse = !ifOp.getElseRegion().empty();
  if (hasElse) {
    return nullptr;
  }

  if (ifOp.getNumResults() > 0) {
    return nullptr;
  }

  auto vecIfOp = ctx.builder.create<scf::IfOp>(loc, TypeRange{}, vecCondition, false);

  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(vecIfOp.thenBlock());

    if (failed(vectorizeRegion(ifOp.getThenRegion(), ctx))) {
      vecIfOp.erase();
      return nullptr;
    }
  }

  ctx.builder.setInsertionPointAfter(vecIfOp);

  return Value();
}

static LogicalResult handleConstantOp(arith::ConstantOp constOp, VectorizationContext &ctx) {
  cloneScalarOp(*constOp.getOperation(), ctx);
  return success();
}

static LogicalResult handleArithOrMathOp(Operation &op, VectorizationContext &ctx) {
  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName != "arith" && dialectName != "math") {
    return success();
  }

  if (op.getNumRegions() != 0) {
    return failure();
  }

  if (op.getNumResults() == 0) {
    return success();
  }

  Value vecValue = vectorizeArithOp(&op, ctx);
  if (!vecValue) {
    return failure();
  }

  ctx.valueMapping.map(op.getResult(0), vecValue);

  return success();
}

static LogicalResult handleIfOp(scf::IfOp ifOp, VectorizationContext &ctx) {
  Value vecIfResult = vectorizeIf(ifOp, ctx);

  if (!vecIfResult && ifOp.getNumResults() > 0) {
    return failure();
  }

  if (vecIfResult) {
    ctx.valueMapping.map(ifOp.getResult(0), vecIfResult);
  }

  return success();
}

static void updateNestedLoopOperands(scf::ForOp nestedForOp, VectorizationContext &ctx) {
  Value mappedLB = ctx.valueMapping.lookupOrDefault(nestedForOp.getLowerBound());
  Value mappedUB = ctx.valueMapping.lookupOrDefault(nestedForOp.getUpperBound());
  Value mappedStep = ctx.valueMapping.lookupOrDefault(nestedForOp.getStep());

  if (mappedLB != nestedForOp.getLowerBound()) {
    nestedForOp.getLowerBoundMutable().assign(mappedLB);
  }
  if (mappedUB != nestedForOp.getUpperBound()) {
    nestedForOp.getUpperBoundMutable().assign(mappedUB);
  }
  if (mappedStep != nestedForOp.getStep()) {
    nestedForOp.getStepMutable().assign(mappedStep);
  }

  for (unsigned i = 0; i < nestedForOp.getNumRegionIterArgs(); ++i) {
    Value initArg = nestedForOp.getInitArgs()[i];
    Value mappedInit = ctx.valueMapping.lookupOrDefault(initArg);
    if (mappedInit != initArg) {
      nestedForOp.getInitArgsMutable()[i].set(mappedInit);
    }
  }
}

static void registerNestedLoopResults(scf::ForOp nestedForOp, VectorizationContext &ctx) {
  Block *insertBlock = ctx.builder.getInsertionBlock();
  if (insertBlock && !insertBlock->empty()) {
    for (auto it = insertBlock->rbegin(); it != insertBlock->rend(); ++it) {
      if (auto vecForOp = dyn_cast<scf::ForOp>(&*it)) {
        for (auto [scalarResult, vecResult] :
             llvm::zip(nestedForOp.getResults(), vecForOp.getResults())) {
          ctx.valueMapping.map(scalarResult, vecResult);
        }
        break;
      }
    }
  }
}

static LogicalResult handleNestedForOp(scf::ForOp nestedForOp, VectorizationContext &ctx) {
  if (ctx.loopToVectorDim.count(nestedForOp)) {
    if (failed(vectorizeLoopMultiDim(nestedForOp, ctx))) {
      cloneScalarOp(*nestedForOp.getOperation(), ctx);
    }
    return success();
  }

  int64_t nestedMaxStep = -1;
  VectorizationMode nestedMode = getVectorizationMode(nestedForOp, nestedMaxStep);

  if (nestedMode == VectorizationMode::None) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  if (ctx.getRank() >= 1 && nestedMode == VectorizationMode::Elementwise &&
      ctx.mode == VectorizationMode::Elementwise) {
    if (failed(vectorizeLoopMultiDim(nestedForOp, ctx))) {
      cloneScalarOp(*nestedForOp.getOperation(), ctx);
    }
    return success();
  }

  int64_t nestedActualStep;
  Value nestedVectorSizeValue;
  Value nestedMaxStepValue;
  Value nestedVecAxis;

  if (nestedMode == VectorizationMode::ReductionY ||
      nestedMode == VectorizationMode::Broadcast) {
    nestedActualStep = ctx.getActualStep();
    nestedVectorSizeValue = ctx.getVectorSizeValue();
    nestedMaxStepValue = ctx.getMaxStepValue();
    nestedVecAxis = ctx.getVectorizationAxis();
  } else {
    nestedActualStep = computeStaticVectorSize(nestedForOp, nestedMaxStep);

    OpBuilder attrBuilder(ctx.builder.getContext());
    attrBuilder.setInsertionPoint(nestedForOp);
    nestedMaxStepValue = attrBuilder.create<arith::ConstantIndexOp>(
        nestedForOp.getLoc(), nestedMaxStep);

    if (ctx.isDynamic()) {
      nestedVectorSizeValue = computeDynamicVectorSize(
          nestedForOp, nestedMaxStepValue, attrBuilder, nestedForOp.getLoc());
    } else {
      nestedVectorSizeValue = nullptr;
    }

    nestedVecAxis = nestedForOp.getInductionVar();
  }

  Value outerIVMapping = Value();
  if ((nestedMode == VectorizationMode::ReductionY ||
       nestedMode == VectorizationMode::Broadcast) &&
      nestedVecAxis) {
    outerIVMapping = ctx.valueMapping.lookupOrDefault(nestedVecAxis);

    if (!outerIVMapping || outerIVMapping == nestedVecAxis) {
      cloneScalarOp(*nestedForOp.getOperation(), ctx);
      return success();
    }
  }

  if (!ctx.builder.getInsertionBlock()) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  updateNestedLoopOperands(nestedForOp, ctx);

  if (failed(vectorizeLoop(nestedForOp, nestedActualStep, nestedVectorSizeValue,
                          nestedMaxStepValue, nestedMode, ctx.builder,
                          nestedVecAxis, outerIVMapping))) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  registerNestedLoopResults(nestedForOp, ctx);
  return success();
}

static LogicalResult vectorizeOneOp(Operation &op, VectorizationContext &ctx) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
    Value vecValue = vectorizeLoad(loadOp, ctx);
    if (!vecValue) {
      return failure();
    }
    ctx.valueMapping.map(loadOp.getResult(), vecValue);
    return success();
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
    vectorizeStore(storeOp, ctx);
    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
    return handleConstantOp(constOp, ctx);
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
    return handleIfOp(ifOp, ctx);
  }

  if (auto nestedForOp = dyn_cast<scf::ForOp>(&op)) {
    return handleNestedForOp(nestedForOp, ctx);
  }

  if (shouldCloneScalarOp(op, ctx)) {
    cloneScalarOp(op, ctx);
    return success();
  }

  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName == "arith" || dialectName == "math") {
    return handleArithOrMathOp(op, ctx);
  }

  cloneScalarOp(op, ctx);
  return success();
}

static LogicalResult vectorizeRegion(Region &region, VectorizationContext &ctx) {
  Block *block = &region.front();

  for (Operation &op : block->without_terminator()) {
    if (failed(vectorizeOneOp(op, ctx))) {
      return failure();
    }
  }

  Operation *originalTerminator = block->getTerminator();
  if (auto originalYieldOp = dyn_cast<scf::YieldOp>(originalTerminator)) {
    SmallVector<Value> vecYieldOperands;

    for (Value operand : originalYieldOp.getOperands()) {
      Value vecOperand = ctx.valueMapping.lookupOrNull(operand);

      if (!vecOperand || !mlir::isa<npuvector::NPUVectorType>(vecOperand.getType())) {
        Value valueToBroadcast = vecOperand ? vecOperand : operand;

        vecOperand = vectorizeBroadcastScalar(valueToBroadcast, ctx);
        if (!vecOperand) {
          return failure();
        }
        ctx.valueMapping.map(operand, vecOperand);
      }
      vecYieldOperands.push_back(vecOperand);
    }

    Block *currentBlock = ctx.builder.getInsertionBlock();
    if (!currentBlock) {
      return failure();
    }

    Operation *autoTerminator = currentBlock->getTerminator();
    if (auto autoYieldOp = dyn_cast<scf::YieldOp>(autoTerminator)) {
      ctx.builder.setInsertionPoint(autoYieldOp);
      ctx.builder.create<scf::YieldOp>(originalYieldOp.getLoc(), vecYieldOperands);
      autoYieldOp.erase();
    } else {
      ctx.builder.setInsertionPointToEnd(currentBlock);
      ctx.builder.create<scf::YieldOp>(originalYieldOp.getLoc(), vecYieldOperands);
    }
  }

  return success();
}

static LogicalResult vectorizeLoopBody(VectorizationContext &ctx) {
  if (ctx.vecLoop) {
    ctx.builder.setInsertionPointToStart(ctx.vecLoop.getBody());
    ctx.valueMapping.map(ctx.scalarLoop.getInductionVar(),
                         ctx.vecLoop.getInductionVar());
  }

  if (!ctx.loopToVectorDim.count(ctx.scalarLoop)) {
    ctx.loopToVectorDim[ctx.scalarLoop] = 0;
  }

  if (ctx.mode == VectorizationMode::ReductionX ||
      ctx.mode == VectorizationMode::ReductionY) {
    for (auto [scalarArg, vecArg] : llvm::zip(
            ctx.scalarLoop.getRegionIterArgs(),
            ctx.vecLoop.getRegionIterArgs())) {
      ctx.valueMapping.map(scalarArg, vecArg);
    }
  }

  auto bodyOps = ctx.scalarLoop.getBody()->without_terminator();
  SmallVector<Operation *> opsToVectorize = llvm::map_to_vector(
      bodyOps, [](Operation &op) { return &op; });

  SmallVector<scf::ForOp> nestedLoopsToErase;

  for (Operation *op : opsToVectorize) {
    if (!op || op->getBlock() == nullptr) {
      continue;
    }

    if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
      int64_t nestedMaxStep = -1;
      VectorizationMode nestedMode = getVectorizationMode(nestedFor, nestedMaxStep);
      if (nestedMode != VectorizationMode::None) {
        nestedLoopsToErase.push_back(nestedFor);
      }
    }

    if (failed(vectorizeOneOp(*op, ctx))) {
      return failure();
    }
  }

  for (auto nestedLoop : nestedLoopsToErase) {
    if (nestedLoop && nestedLoop->getBlock()) {
      nestedLoop.erase();
    }
  }

  ensureReductionYield(ctx);
  return success();
}

static void ensureReductionYield(VectorizationContext &ctx) {
  if (!ctx.vecLoop || !ctx.scalarLoop.getInitArgs().empty()) return;
  if (ctx.mode != VectorizationMode::ReductionX &&
      ctx.mode != VectorizationMode::ReductionY)
    return;
  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) return;
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(ctx.vecLoop.getLoc());
}

static LogicalResult vectorizeElementwiseInline(VectorizationContext &ctx) {
  ctx.builder.setInsertionPoint(ctx.scalarLoop);

  if (!ctx.loopToVectorDim.count(ctx.scalarLoop)) {
    ctx.loopToVectorDim[ctx.scalarLoop] = 0;
  }

  Value iv = ctx.scalarLoop.getInductionVar();
  Value lb = ctx.scalarLoop.getLowerBound();

  ctx.valueMapping.map(iv, lb);

  if (failed(vectorizeLoopBody(ctx))) {
    return failure();
  }

  Operation *parent = ctx.scalarLoop->getParentOp();
  bool isNested = false;
  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) {
        isNested = true;
        break;
      }
    }
    parent = parent->getParentOp();
  }

  if (!isNested) {
    ctx.scalarLoop.erase();
  }

  return success();
}

static void finalizeElementwise(VectorizationContext &ctx) {
  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    body->back().erase();
  }
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(ctx.vecLoop.getLoc());

  Operation *parent = ctx.scalarLoop->getParentOp();
  bool isNested = false;
  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) {
        isNested = true;
        break;
      }
    }
    parent = parent->getParentOp();
  }

  if (!isNested) {
    ctx.scalarLoop.erase();
  }
}

static Value combineReductionResults(OpBuilder &builder, Location loc,
                                     Value lhs, Value rhs, arith::AtomicRMWKind kind) {
  switch (kind) {
    case arith::AtomicRMWKind::addf:
      return builder.create<arith::AddFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::mulf:
      return builder.create<arith::MulFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::maximumf:
      return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::minimumf:
      return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
    default:
      return lhs;
  }
}

static SmallVector<Value> buildTailIndices(memref::LoadOp loadOp, Value inductionVar,
                                           Value tailStart, OpBuilder &builder) {
  SmallVector<Value> indices;
  for (Value idx : loadOp.getIndices()) {
    if (idx == inductionVar) {
      indices.push_back(tailStart);
    } else if (auto constOp = idx.getDefiningOp<arith::ConstantOp>()) {
      indices.push_back(builder.create<arith::ConstantOp>(
          constOp.getLoc(), constOp.getValue()));
    } else {
      indices.push_back(idx);
    }
  }
  return indices;
}

static bool isReductionOp(Operation &op) {
  if (op.hasAttr("reduction_type")) {
    return true;
  }

  if (op.hasAttr("reduction_axes")) {
    return true;
  }

  return false;
}

struct TailVectorTypeInfo {
  npuvector::NPUVectorType vecType;
  SmallVector<Value> dynamicSizes;
};

static TailVectorTypeInfo createTailVectorType(Value tailSize, Type elemType) {
  TailVectorTypeInfo info;
  if (auto tailSizeConst = tailSize.getDefiningOp<arith::ConstantIndexOp>()) {
    info.vecType = npuvector::NPUVectorType::get({tailSizeConst.value()}, elemType);
  } else {
    info.vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, elemType);
    info.dynamicSizes.push_back(tailSize);
  }
  return info;
}

static void vectorizeTailOps(VectorizationContext &tailCtx, VectorizationContext &ctx,
                             Value vecLoopUb, Value tailSize) {
  Location loc = ctx.scalarLoop.getLoc();

  memref::LoadOp tailLoadOp = nullptr;

  tailCtx.valueMapping.map(ctx.scalarLoop.getInductionVar(), vecLoopUb);

  for (Operation &op : ctx.scalarLoop.getBody()->without_terminator()) {
    if (isReductionOp(op)) {
      continue;
    }

    if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
      vectorizeStore(storeOp, tailCtx);
      continue;
    }

    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      tailLoadOp = loadOp;

      Value mappedMemRef = tailCtx.valueMapping.lookupOrDefault(loadOp.getMemRef());

      if (loadIndicesAreLoopInvariant(loadOp, ctx)) {
        SmallVector<Value> indices = llvm::map_to_vector(
            loadOp.getIndices(), [&](Value idx) {
              return tailCtx.valueMapping.lookupOrDefault(idx);
            });

        Value scalarLoad = tailCtx.builder.create<memref::LoadOp>(
            loc, mappedMemRef, indices);

        Value vecValue = vectorizeBroadcastScalar(scalarLoad, tailCtx);
        tailCtx.valueMapping.map(loadOp.getResult(), vecValue);
        continue;
      }

      SmallVector<Value> tailIndices = buildTailIndices(
          loadOp, ctx.scalarLoop.getInductionVar(), vecLoopUb, tailCtx.builder);

      Type memrefElemType = loadOp.getMemRefType().getElementType();
      auto typeInfo = createTailVectorType(tailSize, memrefElemType);

      Value padding = tailCtx.builder.create<arith::ConstantOp>(
          loc, tailCtx.builder.getZeroAttr(memrefElemType));

      SmallVector<Value> tailMaxSizes;
      tailMaxSizes.resize(typeInfo.dynamicSizes.size(), tailSize);
      auto tailRead = tailCtx.builder.create<npuvector::TransferReadOp>(
          loc, typeInfo.vecType, mappedMemRef,
          ValueRange(tailIndices), padding, Value(),
          ValueRange(typeInfo.dynamicSizes), ValueRange(tailMaxSizes));

      tailCtx.valueMapping.map(loadOp.getResult(), tailRead.getResult());
      continue;
    }

    bool isIndexConst = tailLoadOp && llvm::any_of(tailLoadOp.getIndices(), [&](Value idx) {
      return op.getNumResults() > 0 && op.getResult(0) == idx;
    });
    if (isIndexConst) continue;

    if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
      Value scalarConst = tailCtx.builder.create<arith::ConstantOp>(
          constOp.getLoc(), constOp.getValue());
      tailCtx.valueMapping.map(constOp.getResult(), scalarConst);
    } else if (op.getDialect()->getNamespace() == "arith" ||
               op.getDialect()->getNamespace() == "math") {
      Value vecValue = vectorizeArithOp(&op, tailCtx);
      if (vecValue && op.getNumResults() > 0) {
        tailCtx.valueMapping.map(op.getResult(0), vecValue);
      }
    }
  }
}

static Value findValueToReduce(VectorizationContext &tailCtx, VectorizationContext &ctx) {
  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  Value scalarYieldValue = scalarYield.getOperand(0);

  if (auto defOp = scalarYieldValue.getDefiningOp()) {
    if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp>(defOp)) {
      for (Value operand : defOp->getOperands()) {
        if (operand != ctx.scalarLoop.getRegionIterArgs()[0]) {
          Value mappedValue = tailCtx.valueMapping.lookupOrNull(operand);
          if (mappedValue && mlir::isa<npuvector::NPUVectorType>(mappedValue.getType())) {
            return mappedValue;
          }
        }
      }
    }
  }

  return Value();
}

static Value processTailBlock(VectorizationContext &ctx, Value reduced,
                              Value vecLoopUb, Value tailSize, Type elemType,
                              vector::CombiningKind combiningKind) {
  Location loc = ctx.vecLoop.getLoc();

  int64_t tailActualStep;
  Value tailVectorSizeValue;
  if (auto tailSizeConst = tailSize.getDefiningOp<arith::ConstantIndexOp>()) {
    tailActualStep = tailSizeConst.value();
    tailVectorSizeValue = nullptr;
  } else {
    tailActualStep = ctx.getActualStep();
    tailVectorSizeValue = tailSize;
  }

  VectorizationContext tailCtx(ctx.builder, tailActualStep, tailVectorSizeValue,
                               ctx.getMaxStepValue(), ctx.mode, ctx.scalarLoop);
  vectorizeTailOps(tailCtx, ctx, vecLoopUb, tailSize);

  Value valueToReduce = findValueToReduce(tailCtx, ctx);
  if (!valueToReduce) {
    llvm_unreachable("Failed to find value to reduce in tail block");
  }

  auto tailReductionOp = tailCtx.builder.create<npuvector::ReductionOp>(
      loc, elemType, combiningKind, valueToReduce,
      Value(), arith::FastMathFlags::none);
  Value tailReduced = tailReductionOp.getDest();

  return combineReductionResults(tailCtx.builder, loc, reduced, tailReduced, *ctx.reductionKind);
}

static LogicalResult createVectorizedLoop(VectorizationContext &ctx) {
  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) {
    return failure();
  }

  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }

  return success();
}

static void finalizeReductionY(VectorizationContext &ctx) {
  Location loc = ctx.vecLoop.getLoc();

  if (ctx.scalarLoop.getInitArgs().empty()) {
    Block *body = ctx.vecLoop.getBody();
    if (body->empty() || !isa<scf::YieldOp>(body->back())) {
      ctx.builder.setInsertionPointToEnd(body);
      ctx.builder.create<scf::YieldOp>(loc);
    }

    ctx.builder.setInsertionPointAfter(ctx.vecLoop);

    Operation *parent = ctx.scalarLoop->getParentOp();
    bool isNested = false;
    while (parent) {
      if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
        if (hasVectorizationAttr(parentFor)) {
          isNested = true;
          break;
        }
      }
      parent = parent->getParentOp();
    }

    if (!isNested) {
      ctx.scalarLoop.erase();
    }
    return;
  }

  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  Value scalarYieldValue = scalarYield.getOperand(0);
  Value vecYieldValue = ctx.valueMapping.lookupOrNull(scalarYieldValue);

  if (!vecYieldValue) {
    return;
  }

  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    body->back().erase();
  }
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(loc, vecYieldValue);

  ctx.builder.setInsertionPointAfter(ctx.vecLoop);

  Value vectorResult = ctx.vecLoop.getResult(0);
  Value scalarResult = ctx.scalarLoop.getResult(0);

  if (!scalarResult.use_empty()) {
    scalarResult.replaceAllUsesWith(vectorResult);
  }

  Operation *parent = ctx.scalarLoop->getParentOp();
  bool isNested = false;
  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) {
        isNested = true;
        break;
      }
    }
    parent = parent->getParentOp();
  }

  if (!isNested) {
    ctx.scalarLoop.erase();
  }
}

static void finalizeReduction(VectorizationContext &ctx) {
  Location loc = ctx.vecLoop.getLoc();

  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  Value scalarYieldValue = scalarYield.getOperand(0);
  Value vecYieldValue = ctx.valueMapping.lookupOrNull(scalarYieldValue);

  if (!vecYieldValue) {
    return;
  }

  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    body->back().erase();
  }
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(loc, vecYieldValue);

  ctx.builder.setInsertionPointAfter(ctx.vecLoop);

  vector::CombiningKind combiningKind = convertToCombiningKind(*ctx.reductionKind);

  Value vectorAcc = ctx.vecLoop.getResult(0);

  auto npuVectorType = mlir::cast<npuvector::NPUVectorType>(vectorAcc.getType());
  Type elemType = npuVectorType.getElementType();

  auto reductionOp = ctx.builder.create<npuvector::ReductionOp>(
      loc,
      elemType,
      combiningKind,
      vectorAcc,
      Value(),
      arith::FastMathFlags::none);

  Value reduced = reductionOp.getDest();

  Value originalUb = ctx.scalarLoop.getUpperBound();
  Value vecLoopUb = ctx.vecLoop.getUpperBound();
  Value finalResult = reduced;

  if (!ctx.isDynamic()) {
    auto originalUbConst = originalUb.getDefiningOp<arith::ConstantIndexOp>();
    auto lbConst = ctx.scalarLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    bool needTail = false;
    Value tailSize = Value();

    if (originalUbConst && lbConst) {
      int64_t tripCount = originalUbConst.value() - lbConst.value();

      int64_t remainder = tripCount % ctx.getActualStep();
      needTail = (remainder != 0);

      if (needTail) {
        tailSize = ctx.builder.create<arith::ConstantIndexOp>(loc, remainder);
      }
    }

    if (needTail) {
      finalResult = processTailBlock(ctx, reduced, vecLoopUb, tailSize, elemType, combiningKind);
    }
  } else {
    Value lowerBound = ctx.scalarLoop.getLowerBound();

    Value tripCount = ctx.builder.create<arith::SubIOp>(loc, originalUb, lowerBound);

    Value remainder = ctx.builder.create<arith::RemSIOp>(loc, tripCount, ctx.getVectorSizeValue());

    Value c0 = ctx.builder.create<arith::ConstantIndexOp>(loc, 0);

    Value needTail = ctx.builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, remainder, c0);

    auto ifOp = ctx.builder.create<scf::IfOp>(
        loc, TypeRange{elemType}, needTail, /*hasElse=*/true);
    {
      OpBuilder::InsertionGuard guard(ctx.builder);
      ctx.builder.setInsertionPointToStart(ifOp.thenBlock());

      Value mergedResult = processTailBlock(ctx, reduced, vecLoopUb, remainder, elemType, combiningKind);

      ctx.builder.create<scf::YieldOp>(loc, mergedResult);
    }

    {
      OpBuilder::InsertionGuard guard(ctx.builder);
      ctx.builder.setInsertionPointToStart(ifOp.elseBlock());

      ctx.builder.create<scf::YieldOp>(loc, reduced);
    }

    finalResult = ifOp.getResult(0);
  }

  if (!isNeutralElement(*ctx.reductionKind, ctx.origInit, ctx.builder)) {
    finalResult = combineReductionResults(ctx.builder, loc, finalResult, ctx.origInit, *ctx.reductionKind);
  }

  Value scalarResult = ctx.scalarLoop.getResult(0);

  if (!scalarResult.use_empty()) {
    scalarResult.replaceAllUsesWith(finalResult);
  }

  Operation *parent = ctx.scalarLoop->getParentOp();
  bool isNested = false;
  while (parent) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) {
        isNested = true;
        break;
      }
    }
    parent = parent->getParentOp();
  }

  if (!isNested) {
    ctx.scalarLoop.erase();
  }
}

static void collectVectorizationStrategy(
    scf::ForOp rootLoop,
    VectorizationContext &ctx,
    OpBuilder &attrBuilder,
    unsigned &nextDim) {

  int64_t maxStepFromAttr = -1;
  VectorizationMode mode = getVectorizationMode(rootLoop, maxStepFromAttr);

  if (mode == VectorizationMode::None || maxStepFromAttr <= 0) return;
  if (mode == VectorizationMode::ReductionY) return;
  if (mode == VectorizationMode::Broadcast) return;

  auto [skip, isDynamic] = checkLoopEligibility(rootLoop);
  if (skip) return;

  unsigned dim = nextDim++;

  int64_t actualStep = computeStaticVectorSize(rootLoop, maxStepFromAttr);
  Value maxStepValue = attrBuilder.create<arith::ConstantIndexOp>(
      rootLoop.getLoc(), maxStepFromAttr);

  Value vectorSizeValue;
  if (isDynamic) {
    attrBuilder.setInsertionPoint(rootLoop);
    vectorSizeValue = computeDynamicVectorSize(
        rootLoop, maxStepValue, attrBuilder, rootLoop.getLoc());
  }

  if (dim >= ctx.vectorSizes.size()) {
    ctx.vectorSizes.push_back(actualStep);
    ctx.vectorSizeValues.push_back(vectorSizeValue);
    ctx.maxStepValues.push_back(maxStepValue);
  } else {
    ctx.vectorSizes[dim] = actualStep;
    ctx.vectorSizeValues[dim] = vectorSizeValue;
    ctx.maxStepValues[dim] = maxStepValue;
  }

  ctx.loopToVectorDim[rootLoop] = dim;

  rootLoop.getBody()->walk([&](scf::ForOp nestedLoop) {
    if (nestedLoop == rootLoop) return;
    if (nestedLoop->getParentOfType<scf::ForOp>() != rootLoop) return;
    if (hasVectorizationAttr(nestedLoop)) {
      attrBuilder.setInsertionPoint(nestedLoop);
      collectVectorizationStrategy(nestedLoop, ctx, attrBuilder, nextDim);
    }
  });
}

static LogicalResult vectorizeLoopMultiDim(scf::ForOp nestedLoop,
                                           VectorizationContext &parentCtx) {
  if (parentCtx.loopToVectorDim.count(nestedLoop)) {
    unsigned dim = parentCtx.loopToVectorDim.lookup(nestedLoop);
    int64_t nestedActualStep = parentCtx.vectorSizes[dim];
    Location loc = nestedLoop.getLoc();

    Value lb = parentCtx.valueMapping.lookupOrDefault(nestedLoop.getLowerBound());
    Value ub = parentCtx.valueMapping.lookupOrDefault(nestedLoop.getUpperBound());

    bool needLoop = false;
    if (parentCtx.isDynamic()) {
      needLoop = true;
    } else {
      auto ubConst = nestedLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
      auto lbConst = nestedLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
      if (ubConst && lbConst) {
        int64_t tripCount = ubConst.value() - lbConst.value();
        needLoop = (nestedActualStep < tripCount);
      }
    }

    Value nestedStepValue;
    if (needLoop) {
      if (parentCtx.vectorSizeValues[dim]) {
        // Recompute in current block using MAPPED lb/ub: nestedLoop.getUpperBound()
        // returns original SSA from the block that gets erased. Using mapped values
        // avoids dangling refs and verifier crash (Type::isSignlessIntOrIndex on
        // invalid operand).
        Value maxStep = parentCtx.builder.create<arith::ConstantIndexOp>(
            loc, nestedActualStep);
        nestedStepValue = computeDynamicVectorSize(
            ub, lb, maxStep, parentCtx.builder, loc);
      } else {
        nestedStepValue =
            parentCtx.builder.create<arith::ConstantIndexOp>(loc, nestedActualStep);
      }
      scf::ForOp innerVecLoop = parentCtx.builder.create<scf::ForOp>(
          nestedLoop.getLoc(), lb, ub, nestedStepValue, std::nullopt,
          [](OpBuilder &, Location, Value, ValueRange) {});
      parentCtx.valueMapping.map(nestedLoop.getInductionVar(),
                                 innerVecLoop.getInductionVar());
      parentCtx.builder.setInsertionPointToStart(innerVecLoop.getBody());

      for (Operation &op : nestedLoop.getBody()->without_terminator()) {
        if (failed(vectorizeOneOp(op, parentCtx))) {
          return failure();
        }
      }

      parentCtx.builder.create<scf::YieldOp>(nestedLoop.getLoc());
      parentCtx.builder.setInsertionPointAfter(innerVecLoop);
    } else {
      parentCtx.valueMapping.map(nestedLoop.getInductionVar(), lb);
      for (Operation &op : nestedLoop.getBody()->without_terminator()) {
        if (failed(vectorizeOneOp(op, parentCtx))) {
          return failure();
        }
      }
    }
    return success();
  }

  int64_t nestedMaxStep = -1;
  VectorizationMode nestedMode = getVectorizationMode(nestedLoop, nestedMaxStep);
  if (nestedMode != VectorizationMode::Elementwise || nestedMaxStep <= 0)
    return failure();

  auto [skip, isDynamic] = checkLoopEligibility(nestedLoop);
  if (skip) return failure();

  int64_t nestedActualStep = computeStaticVectorSize(nestedLoop, nestedMaxStep);

  OpBuilder attrBuilder(parentCtx.builder.getContext());
  attrBuilder.setInsertionPoint(nestedLoop);
  Value nestedMaxStepValue = attrBuilder.create<arith::ConstantIndexOp>(
      nestedLoop.getLoc(), nestedMaxStep);

  Value nestedVectorSizeValue;
  if (isDynamic) {
    nestedVectorSizeValue = computeDynamicVectorSize(
        nestedLoop, nestedMaxStepValue, attrBuilder, nestedLoop.getLoc());
  }

  unsigned newDim = parentCtx.vectorSizes.size();
  parentCtx.vectorSizes.push_back(nestedActualStep);
  parentCtx.vectorSizeValues.push_back(nestedVectorSizeValue);
  parentCtx.maxStepValues.push_back(nestedMaxStepValue);
  parentCtx.loopToVectorDim[nestedLoop] = newDim;

  Value lb = nestedLoop.getLowerBound();
  Value mappedLB = parentCtx.valueMapping.lookupOrDefault(lb);
  parentCtx.valueMapping.map(nestedLoop.getInductionVar(), mappedLB);

  for (Operation &op : nestedLoop.getBody()->without_terminator()) {
    if (failed(vectorizeOneOp(op, parentCtx))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult applyNestedVecAxisMapping(VectorizationMode mode, Value vecAxis,
                                               Value outerIVMapping,
                                               VectorizationContext &ctx) {
  if ((mode != VectorizationMode::ReductionY &&
       mode != VectorizationMode::Broadcast) ||
      !vecAxis || !outerIVMapping)
    return success();
  if (outerIVMapping != vecAxis) {
    ctx.valueMapping.map(vecAxis, outerIVMapping);
    return success();
  }
  return failure();
}

static LogicalResult createVecLoopAndPopulateBody(VectorizationContext &ctx,
                                                  scf::ForOp scalarLoop) {
  if (!ctx.builder.getInsertionBlock()) {
    ctx.builder.setInsertionPoint(scalarLoop);
  }
  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop)
    return failure();
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }
  return success();
}

static LogicalResult vectorizeLoopElementwiseCase(VectorizationContext &ctx,
                                                  scf::ForOp scalarLoop,
                                                  int64_t actualStep) {
  bool needLoop = false;
  if (ctx.isDynamic()) {
    needLoop = true;
  } else {
    auto ubConst = scalarLoop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lbConst = scalarLoop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    if (ubConst && lbConst) {
      int64_t tripCount = ubConst.value() - lbConst.value();
      needLoop = (actualStep < tripCount);
    }
  }
  if (needLoop) {
    ctx.builder.setInsertionPoint(scalarLoop);
    if (failed(createVectorizedLoop(ctx)))
      return failure();
    finalizeElementwise(ctx);
    return success();
  }
  return vectorizeElementwiseInline(ctx);
}

static LogicalResult vectorizeLoopReductionXCase(VectorizationContext &ctx,
                                                 scf::ForOp scalarLoop) {
  ctx.builder.setInsertionPoint(scalarLoop);
  if (failed(createVectorizedLoop(ctx)))
    return failure();
  finalizeReduction(ctx);
  return success();
}

static LogicalResult vectorizeLoopReductionYCase(VectorizationContext &ctx,
                                                 scf::ForOp scalarLoop) {
  if (failed(createVecLoopAndPopulateBody(ctx, scalarLoop)))
    return failure();
  finalizeReductionY(ctx);
  return success();
}

static LogicalResult vectorizeLoopBroadcastCase(VectorizationContext &ctx,
                                                scf::ForOp scalarLoop) {
  if (failed(createVecLoopAndPopulateBody(ctx, scalarLoop)))
    return failure();
  finalizeElementwise(ctx);
  return success();
}

static LogicalResult vectorizeLoop(
    scf::ForOp scalarLoop,
    int64_t actualStep,
    Value vfValue,
    Value maxStepValue,
    VectorizationMode mode,
    OpBuilder &builder,
    Value vecAxis,
    Value outerIVMapping) {

  VectorizationContext ctx(builder, actualStep, vfValue, maxStepValue, mode, scalarLoop, vecAxis);

  ctx.loopToVectorDim[scalarLoop] = 0;

  if (failed(applyNestedVecAxisMapping(mode, vecAxis, outerIVMapping, ctx)))
    return failure();

  switch (mode) {
  case VectorizationMode::Elementwise:
    return vectorizeLoopElementwiseCase(ctx, scalarLoop, actualStep);
  case VectorizationMode::ReductionX:
    return vectorizeLoopReductionXCase(ctx, scalarLoop);
  case VectorizationMode::ReductionY:
    return vectorizeLoopReductionYCase(ctx, scalarLoop);
  case VectorizationMode::Broadcast:
    return vectorizeLoopBroadcastCase(ctx, scalarLoop);
  default:
    return failure();
  }
}

static bool hasNestedElementwiseVecLoop(scf::ForOp loop) {
  bool found = false;
  loop.getBody()->walk([&](scf::ForOp nestedLoop) {
    if (nestedLoop != loop && hasVectorizationAttr(nestedLoop)) {
      int64_t nestedMaxStep = -1;
      VectorizationMode nestedMode = getVectorizationMode(nestedLoop, nestedMaxStep);
      if (nestedMode == VectorizationMode::Elementwise) found = true;
    }
  });
  return found;
}

static void runMultiDimPath(scf::ForOp loop, VectorizationMode mode,
                           OpBuilder &builder, MLIRContext *context,
                           int64_t maxStepFromAttr, bool isDynamic) {
  OpBuilder attrBuilder(context);
  attrBuilder.setInsertionPoint(loop);
  SmallVector<int64_t> vecSizes;
  SmallVector<Value> vecSizeValues;
  SmallVector<Value> maxSteps;
  VectorizationContext ctx(builder, vecSizes, vecSizeValues, maxSteps,
                          mode, loop, loop.getInductionVar());
  unsigned nextDim = 0;
  collectVectorizationStrategy(loop, ctx, attrBuilder, nextDim);
  if (ctx.vectorSizes.empty()) return;
  ctx.builder.setInsertionPoint(loop);
  bool needLoop = false;
  if (ctx.isDynamic()) {
    needLoop = true;
  } else {
    auto ubConst = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto lbConst = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    if (ubConst && lbConst)
      needLoop = (ctx.vectorSizes[0] < ubConst.value() - lbConst.value());
  }
  if (needLoop) {
    if (failed(createVectorizedLoop(ctx))) return;
    finalizeElementwise(ctx);
  } else {
    (void)vectorizeElementwiseInline(ctx);
  }
}

static void runNormal1DPath(scf::ForOp loop, VectorizationMode mode,
                            OpBuilder &builder, MLIRContext *context,
                            int64_t maxStepFromAttr, bool isDynamic) {
  OpBuilder attrBuilder(context);
  attrBuilder.setInsertionPoint(loop);
  Value maxStepValue = attrBuilder.create<arith::ConstantIndexOp>(
      loop.getLoc(), maxStepFromAttr);
  Value vectorSizeValue =
      isDynamic ? computeDynamicVectorSize(loop, maxStepValue, attrBuilder,
                                           loop.getLoc())
                : Value();
  int64_t actualStep = computeStaticVectorSize(loop, maxStepFromAttr);
  (void)vectorizeLoop(loop, actualStep, vectorSizeValue, maxStepValue,
                     mode, builder, loop.getInductionVar());
}

class NPUVectorVectorizePass
    : public mlir::scf::impl::NPUVectorVectorizePassBase<NPUVectorVectorizePass> {
 public:
  NPUVectorVectorizePass() = default;
  NPUVectorVectorizePass(const NPUVectorVectorizePass &) = default;

  StringRef getArgument() const override { return "npuvector-vectorize"; }

  StringRef getDescription() const override {
    return "SCF loop vectorization using NPUVector with dynamic shape support";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect,
                    npuvector::NPUVectorDialect,
                    memref::MemRefDialect,
                    func::FuncDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(&getContext());

    SmallVector<scf::ForOp> allCandidateLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (hasVectorizationAttr(forOp)) allCandidateLoops.push_back(forOp);
    });

    SmallVector<scf::ForOp> topLevelLoops;
    for (scf::ForOp loop : allCandidateLoops) {
      bool isNested = false;
      for (Operation *parent = loop->getParentOp();
           parent && !isa<func::FuncOp>(parent);
           parent = parent->getParentOp()) {
        if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
          if (hasVectorizationAttr(parentFor)) { isNested = true; break; }
        }
      }
      if (!isNested) topLevelLoops.push_back(loop);
    }

    for (scf::ForOp loop : topLevelLoops) {
      int64_t maxStepFromAttr = -1;
      VectorizationMode mode = getVectorizationMode(loop, maxStepFromAttr);
      if (mode == VectorizationMode::None || maxStepFromAttr <= 0) continue;

      auto [skip, isDynamic] = checkLoopEligibility(loop);
      if (skip) continue;

      if (hasNestedElementwiseVecLoop(loop) &&
          mode == VectorizationMode::Elementwise) {
        runMultiDimPath(loop, mode, builder, &getContext(), maxStepFromAttr,
                       isDynamic);
      } else {
        runNormal1DPath(loop, mode, builder, &getContext(), maxStepFromAttr,
                       isDynamic);
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::scf::createNPUVectorVectorizePass() {
  return std::make_unique<NPUVectorVectorizePass>();
}
