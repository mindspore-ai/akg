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
//               collectVectorizationStrategy into a multi-dim LoopVectorizationCtx
//               (broadcast/reduction_y inner loops are not collected as vector dims)
//   Vectorization axis (which loop's IV drives the tile / loopToVectorDim):
//     - Loops tagged `vector=N`, `reduction_x`, or `reduction_all`: that loop's IV
//       is the vectorization axis for that nest.
//     - Loops tagged `reduction_y` or `broadcast`: that loop's IV is NOT the axis;
//       the axis is the ancestor `vector=` loop's IV (vectorizationAxis /
//       vectorAxisVectorDim).
//   3. Create and populate vectorized loop
//      ├── vectorizeLoad → memref.load (scalar) when indices carry no vector axis; else
//      │   npuvector.transfer_read (+ transpose). No npuvector.broadcast at load sites.
//      ├── vectorizeStore → npuvector.transfer_write (+ transpose + rank-lift broadcast); scalar
//      │   store values broadcast here: rank = indices that use a vector axis (subset of ctx), else
//      │   full ctx. Vector store: transpose aligns valueDimOrder to store-index order among the
//      │   value's axes (deduped); then `npuvector.broadcast` rank-lifts to full store-index vector
//      │   rank when needed (`resultDimToCtxAxis` = per-result-dim global axis, same as arith peer).
//      ├── vectorizeArithOp → type conversion and arithmetic; scalar operands broadcast here.
//      │   If **every** operand (after `valueMapping`) is non-`!npuvector`, clone the op (stay scalar);
//      │   otherwise broadcast scalar slots to the peer `!npuvector` tile.
//      └── vectorizeBroadcastScalar → npuvector.broadcast (rank-lift `dimension` must resolve; no fallback)
//   4. Finalization
//      ├── Elementwise: inline or loop-based transformation
//      └── Reduction: vector reduction + tail processing + init value merging
//   5. Phase 2 (same pass tail): VF=1 sweep — single fictional LoopVectorizationCtx (step=max=1, Elementwise,
//      no scf.for anchor, vf1FuncLevelNoAnchor) over the whole func for remaining scalar memref.load chains.
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <deque>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "akg/Dialect/SCF/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

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

struct LoopVectorizationCtx {
  OpBuilder &builder;

  LoopVectorizationCtx *parent = nullptr;

  scf::ForOp scalarLoop;
  scf::ForOp vecLoop;
  VectorizationMode mode;
  unsigned localDim = 0;

  SmallVector<int64_t> allVectorSizes;
  SmallVector<Value> allVectorSizeValues;
  SmallVector<Value> allMaxStepValues;
  DenseMap<Operation *, unsigned> allLoopToVectorDim;

  IRMapping valueMapping;
  DenseMap<Value, SmallVector<int>> valueDimOrder;

  /// Per `iter_arg` / loop result for ReductionX/Y (order matches `scalarLoop` init args).
  SmallVector<arith::AtomicRMWKind> reductionKinds;
  SmallVector<Value> origInits;

  Value vectorizationAxis;
  std::optional<unsigned> vectorAxisVectorDim;

  DenseSet<Operation *> absorbedOps;

  DenseMap<Value, Value> allocBypass;

  bool vf1FuncLevelNoAnchor = false;

  LoopVectorizationCtx(OpBuilder &b, SmallVector<int64_t> vecSizes, SmallVector<Value> vecSizeVals,
                       SmallVector<Value> maxVals, VectorizationMode m, scf::ForOp loop, Value vecAxis = nullptr)
      : builder(b),
        scalarLoop(loop),
        vecLoop(nullptr),
        mode(m),
        allVectorSizes(std::move(vecSizes)),
        allVectorSizeValues(std::move(vecSizeVals)),
        allMaxStepValues(std::move(maxVals)),
        vectorizationAxis(vecAxis) {}

  LoopVectorizationCtx(OpBuilder &b, int64_t actualStepVal, Value vfVal, Value maxVal, VectorizationMode m,
                       scf::ForOp loop, Value vecAxis = nullptr)
      : builder(b),
        scalarLoop(loop),
        vecLoop(nullptr),
        mode(m),
        allVectorSizes({actualStepVal}),
        allVectorSizeValues({vfVal}),
        allMaxStepValues({maxVal}),
        vectorizationAxis(vecAxis) {}

  int64_t getRank() const { return allVectorSizes.size(); }

  bool isDynamic() const {
    return llvm::any_of(allVectorSizeValues, [](Value v) { return v != nullptr; });
  }

  int64_t getActualStep() const { return allVectorSizes.back(); }
  Value getVectorSizeValue() const { return allVectorSizeValues.back(); }
  Value getMaxStepValue() const { return allMaxStepValues.back(); }

  Value getVectorizationAxis() {
    if (vectorizationAxis && (mode == VectorizationMode::ReductionY || mode == VectorizationMode::Broadcast)) {
      return vectorizationAxis;
    }
    if (!scalarLoop.getOperation()) return Value();
    return scalarLoop.getInductionVar();
  }

  npuvector::NPUVectorType getVectorType(Type elemType) const {
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < allVectorSizes.size(); ++i) {
      if (allVectorSizeValues[i])
        shape.push_back(ShapedType::kDynamic);
      else
        shape.push_back(allVectorSizes[i]);
    }
    return npuvector::NPUVectorType::get(shape, elemType);
  }

  int getVectorDimForIV(Value iv) const {
    for (auto &[op, dim] : allLoopToVectorDim) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (forOp.getInductionVar() == iv) return static_cast<int>(dim);
      }
    }
    if (vectorizationAxis && iv == vectorizationAxis && vectorAxisVectorDim)
      return static_cast<int>(*vectorAxisVectorDim);
    return -1;
  }

  static LoopVectorizationCtx createChild(LoopVectorizationCtx &parentCtx, scf::ForOp childLoop,
                                          VectorizationMode childMode, int64_t childVecSize, Value childVecSizeVal,
                                          Value childMaxStepVal) {
    SmallVector<int64_t> emptyS;
    SmallVector<Value> emptyV;
    LoopVectorizationCtx child(parentCtx.builder, emptyS, emptyV, emptyV, childMode, childLoop);
    child.parent = &parentCtx;

    SmallVector<std::pair<Operation *, unsigned>> ancestorLoops;
    for (Operation *p = childLoop->getParentOp(); p; p = p->getParentOp()) {
      auto it = parentCtx.allLoopToVectorDim.find(p);
      if (it != parentCtx.allLoopToVectorDim.end()) ancestorLoops.push_back({it->first, it->second});
    }
    llvm::sort(ancestorLoops,
               [](const auto &pairLeft, const auto &pairRight) { return pairLeft.second < pairRight.second; });

    DenseMap<unsigned, unsigned> parentDimToLocalDim;
    for (auto &[loop, parentDim] : ancestorLoops) {
      unsigned lDim = child.allVectorSizes.size();
      parentDimToLocalDim[parentDim] = lDim;
      child.allVectorSizes.push_back(parentCtx.allVectorSizes[parentDim]);
      int64_t maxStepInt = parentCtx.allVectorSizes[parentDim];
      if (auto cst = parentCtx.allMaxStepValues[parentDim].getDefiningOp<arith::ConstantIndexOp>())
        maxStepInt = cst.value();
      child.allMaxStepValues.push_back(
        parentCtx.builder.create<arith::ConstantIndexOp>(childLoop.getLoc(), maxStepInt));
      Value vfVal = nullptr;
      if (parentCtx.allVectorSizeValues[parentDim])
        vfVal = parentCtx.valueMapping.lookupOrDefault(parentCtx.allVectorSizeValues[parentDim]);
      child.allVectorSizeValues.push_back(vfVal);
      child.allLoopToVectorDim[loop] = lDim;
    }

    bool addsVecDim = (childMode == VectorizationMode::Elementwise || childMode == VectorizationMode::ReductionX);
    if (addsVecDim) {
      child.localDim = child.allVectorSizes.size();
      child.allVectorSizes.push_back(childVecSize);
      child.allVectorSizeValues.push_back(childVecSizeVal);
      child.allMaxStepValues.push_back(childMaxStepVal);
      child.allLoopToVectorDim[childLoop] = child.localDim;
    } else {
      child.localDim = std::numeric_limits<unsigned>::max();
      for (auto it = ancestorLoops.rbegin(); it != ancestorLoops.rend(); ++it) {
        auto ancestorFor = dyn_cast<scf::ForOp>(it->first);
        if (ancestorFor && ancestorFor->hasAttr(kVectorAttr)) {
          child.vectorizationAxis = ancestorFor.getInductionVar();
          child.vectorAxisVectorDim = parentDimToLocalDim[it->second];
          break;
        }
      }
    }

    for (const auto &kv : parentCtx.valueMapping.getValueMap()) child.valueMapping.map(kv.first, kv.second);

    for (const auto &kv : parentCtx.valueDimOrder) child.valueDimOrder[kv.first] = kv.second;

    return child;
  }
};

static void mergeAncestorValueDimOrderMissingKeys(LoopVectorizationCtx &ctx);
static Value vectorizeBroadcastScalar(Value scalarVal, LoopVectorizationCtx &ctx,
                                      npuvector::NPUVectorType targetType = {},
                                      llvm::ArrayRef<int> resultDimToCtxAxis = {});
static void ensureReductionYield(LoopVectorizationCtx &ctx);
static LogicalResult vectorizeOneOp(Operation &op, LoopVectorizationCtx &ctx);
static void processLoop(LoopVectorizationCtx &ctx);
static bool definitionGraphContainsValue(Value conditionSsaValue, Value targetSsaValue);

static bool hasVectorizationAttr(Operation *op) {
  return op->hasAttr(kVectorAttr) || op->hasAttr(kBroadcastLoopAttr) || op->hasAttr(kReductionXLoopAttr) ||
         op->hasAttr(kReductionYLoopAttr) || op->hasAttr(kReductionAllLoopAttr);
}

static bool hasIndexResult(Operation &op) {
  return llvm::any_of(op.getResults(), [](Value resultVal) { return resultVal.getType().isIndex(); });
}

static bool shouldCloneScalarOp(Operation &op, LoopVectorizationCtx &ctx) {
  if (isa<affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp>(&op)) {
    return true;
  }
  if (hasIndexResult(op)) {
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

static Value computeDynamicVectorSize(scf::ForOp loop, Value maxStepValue, OpBuilder &builder, Location loc,
                                      const IRMapping *valueMapping = nullptr) {
  Value upperBound = loop.getUpperBound();
  Value lowerBound = loop.getLowerBound();
  if (valueMapping) {
    upperBound = valueMapping->lookupOrDefault(upperBound);
    lowerBound = valueMapping->lookupOrDefault(lowerBound);
  }

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

/// True when `defOp` is a binary op whose lhs/rhs references the reduction carry (`iter_arg`).
static bool binaryReductionUsesIterArg(Operation *defOp, Value iterArg) {
  return defOp && defOp->getNumOperands() >= 2 &&
         (defOp->getOperand(0) == iterArg || defOp->getOperand(1) == iterArg);
}

/// Maps yield-defining binary arith op to `AtomicRMWKind` when it uses `iterArg` as one combine input.
static std::optional<arith::AtomicRMWKind> matchYieldDefToReductionKind(Operation *defOp, Value iterArg) {
  if (!binaryReductionUsesIterArg(defOp, iterArg)) return std::nullopt;

  return llvm::TypeSwitch<Operation *, std::optional<arith::AtomicRMWKind>>(defOp)
    .Case([](arith::AddFOp) { return arith::AtomicRMWKind::addf; })
    .Case([](arith::MulFOp) { return arith::AtomicRMWKind::mulf; })
    .Case([](arith::MaximumFOp) { return arith::AtomicRMWKind::maximumf; })
    .Case([](arith::MinimumFOp) { return arith::AtomicRMWKind::minimumf; })
    .Case([](arith::AddIOp) { return arith::AtomicRMWKind::addi; })
    .Case([](arith::MulIOp) { return arith::AtomicRMWKind::muli; })
    .Case([](arith::MaxSIOp) { return arith::AtomicRMWKind::maxs; })
    .Case([](arith::MinSIOp) { return arith::AtomicRMWKind::mins; })
    .Case([](arith::MaxUIOp) { return arith::AtomicRMWKind::maxu; })
    .Case([](arith::MinUIOp) { return arith::AtomicRMWKind::minu; })
    .Default([](Operation *) -> std::optional<arith::AtomicRMWKind> { return std::nullopt; });
}

static std::optional<arith::AtomicRMWKind> detectReductionKindForOperand(scf::ForOp loop, unsigned resultIndex) {
  if (loop.getInitArgs().empty()) {
    return std::nullopt;
  }

  auto yieldOp = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yieldOp || resultIndex >= yieldOp.getNumOperands()) {
    return std::nullopt;
  }

  Value yieldValue = yieldOp.getOperand(resultIndex);
  Operation *defOp = yieldValue.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }

  return matchYieldDefToReductionKind(defOp, loop.getRegionIterArgs()[resultIndex]);
}

static Value createNeutralValue(arith::AtomicRMWKind kind, Type elemType, LoopVectorizationCtx &ctx, Location loc) {
  Attribute neutralAttr = arith::getIdentityValueAttr(kind, elemType, ctx.builder, loc);

  npuvector::NPUVectorType vecType = ctx.getVectorType(elemType);

  if (ctx.isDynamic()) {
    Value neutralScalar = ctx.builder.create<arith::ConstantOp>(loc, mlir::cast<TypedAttr>(neutralAttr));

    // One (extent, max) pair per result *rank* (not only `?` dims), matching
    // npuvector.transfer_read in `vectorizeLoad` and `allocBroadcastBuffer` when
    // dynamicSizes.size() == npuVecType.getRank().
    SmallVector<Value> dynamicSizes;
    SmallVector<Value> maxSizes;
    for (unsigned i = 0; i < ctx.allVectorSizes.size(); ++i) {
      maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[i]));
      if (ctx.allVectorSizeValues[i])
        dynamicSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[i]));
      else
        dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[i]));
    }
    Value neutralVec =
      ctx.builder.create<npuvector::BroadcastOp>(loc, vecType, neutralScalar, ValueRange(dynamicSizes),
                                                 ValueRange(maxSizes), ctx.builder.getDenseI64ArrayAttr({}));

    return neutralVec;

  } else {
    auto vecAttr = DenseElementsAttr::get(vecType, neutralAttr);
    Value neutralVec = ctx.builder.create<arith::ConstantOp>(loc, vecAttr);

    return neutralVec;
  }
}

static bool isNeutralElement(arith::AtomicRMWKind kind, Value value, OpBuilder &builder) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  Attribute neutralAttr = arith::getIdentityValueAttr(kind, value.getType(), builder, value.getLoc());

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

static std::optional<int64_t> tryConstantIndex(Value indexValue);

static scf::ForOp createEmptyVectorizedLoop(LoopVectorizationCtx &ctx) {
  Location loc = ctx.scalarLoop.getLoc();

  Value newStepValue;
  if (ctx.mode == VectorizationMode::ReductionY || ctx.mode == VectorizationMode::Broadcast) {
    newStepValue = ctx.scalarLoop.getStep();
  } else {
    unsigned dim = ctx.allLoopToVectorDim.lookup(ctx.scalarLoop);
    if (ctx.allVectorSizeValues[dim]) {
      newStepValue = ctx.allVectorSizeValues[dim];
    } else {
      newStepValue = ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[dim]);
    }
  }

  SmallVector<Value> neutralVecs;
  if (ctx.mode == VectorizationMode::ReductionX || ctx.mode == VectorizationMode::ReductionY) {
    if (!ctx.scalarLoop.getInitArgs().empty()) {
      ctx.reductionKinds.clear();
      ctx.origInits.clear();
      for (unsigned idx = 0; idx < ctx.scalarLoop.getNumRegionIterArgs(); ++idx) {
        auto kind = detectReductionKindForOperand(ctx.scalarLoop, idx);
        if (!kind) return nullptr;

        ctx.reductionKinds.push_back(*kind);
        ctx.origInits.push_back(ctx.scalarLoop.getInitArgs()[idx]);

        Type elemType = ctx.scalarLoop.getRegionIterArgs()[idx].getType();
        Value neutralVec = createNeutralValue(*kind, elemType, ctx, loc);
        if (!neutralVec) return nullptr;
        neutralVecs.push_back(neutralVec);
      }
    }
  }

  Value upperBound = ctx.scalarLoop.getUpperBound();
  Value lowerBound = ctx.scalarLoop.getLowerBound();

  if (ctx.mode == VectorizationMode::ReductionX) {
    unsigned dim = ctx.allLoopToVectorDim.lookup(ctx.scalarLoop);
    std::optional<int64_t> ubOpt = tryConstantIndex(upperBound);
    std::optional<int64_t> lbOpt = tryConstantIndex(lowerBound);
    if (ubOpt && lbOpt) {
      int64_t lbVal = *lbOpt;
      int64_t ubVal = *ubOpt;
      int64_t tripCount = ubVal - lbVal;
      int64_t alignedTripCount = (tripCount / ctx.getActualStep()) * ctx.getActualStep();
      int64_t alignedUb = lbVal + alignedTripCount;
      if (alignedUb < ubVal) {
        upperBound = ctx.builder.create<arith::ConstantIndexOp>(loc, alignedUb);
      }
    } else {
      Value vfAlign = ctx.allVectorSizeValues[dim];
      if (!vfAlign) vfAlign = ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.getActualStep());
      Value tripCount = ctx.builder.create<arith::SubIOp>(loc, upperBound, lowerBound);
      Value numIterations = ctx.builder.create<arith::DivSIOp>(loc, tripCount, vfAlign);
      Value alignedTripCount = ctx.builder.create<arith::MulIOp>(loc, numIterations, vfAlign);
      upperBound = ctx.builder.create<arith::AddIOp>(loc, alignedTripCount, lowerBound);
    }
  }

  scf::ForOp vecLoop;
  if (!neutralVecs.empty()) {
    vecLoop = ctx.builder.create<scf::ForOp>(loc, lowerBound, upperBound, newStepValue, ValueRange(neutralVecs),
                                             [](OpBuilder &, Location, Value, ValueRange) {});
  } else {
    vecLoop = ctx.builder.create<scf::ForOp>(loc, lowerBound, upperBound, newStepValue, std::nullopt,
                                             [](OpBuilder &, Location, Value, ValueRange) {});
  }

  return vecLoop;
}

static std::optional<int64_t> tryConstantIndex(Value indexValue) {
  if (auto constantIndexOp = indexValue.getDefiningOp<arith::ConstantIndexOp>()) return constantIndexOp.value();
  if (auto constantOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
    if (!indexValue.getType().isIndex()) return std::nullopt;
    if (auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) return integerAttr.getValue().getSExtValue();
  }
  return std::nullopt;
}

static bool gatherVectorDynExtents(Value vectorVal, SmallVectorImpl<Value> &outExtents) {
  Operation *defOp = vectorVal.getDefiningOp();
  if (!defOp) return false;

  if (auto readOp = mlir::dyn_cast<npuvector::TransferReadOp>(defOp)) {
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vectorVal.getType());
    if (!nvt) return false;
    npuvector::TransferReadOp::Adaptor readAdaptor(readOp);
    ValueRange dynSizes = readAdaptor.getDynamicSizes();
    if (static_cast<int64_t>(dynSizes.size()) != static_cast<int64_t>(nvt.getRank())) return false;
    outExtents.assign(dynSizes.begin(), dynSizes.end());
    return true;
  }

  if (auto transposeOp = mlir::dyn_cast<npuvector::TransposeOp>(defOp)) {
    SmallVector<Value> innerExtents;
    if (!gatherVectorDynExtents(transposeOp.getVector(), innerExtents)) return false;
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto resultNvt = mlir::dyn_cast<npuvector::NPUVectorType>(vectorVal.getType());
    if (!resultNvt || perm.size() != static_cast<size_t>(resultNvt.getRank())) return false;
    const unsigned rank = static_cast<unsigned>(resultNvt.getRank());
    outExtents.resize(rank);
    for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
      const int64_t srcDim = perm[resultDim];
      if (srcDim < 0 || static_cast<size_t>(srcDim) >= innerExtents.size()) return false;
      outExtents[resultDim] = innerExtents[static_cast<unsigned>(srcDim)];
    }
    return true;
  }

  return false;
}

static std::optional<unsigned> matchExtentValueToCtxAxis(Value extent, const LoopVectorizationCtx &ctx) {
  Value mappedExt = ctx.valueMapping.lookupOrDefault(extent);
  llvm::SmallVector<unsigned, 4> hits;
  for (unsigned axisIdx = 0; axisIdx < ctx.allVectorSizes.size(); ++axisIdx) {
    if (ctx.allVectorSizeValues[axisIdx]) {
      Value refLen = ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[axisIdx]);
      if (mappedExt == refLen || extent == refLen) hits.push_back(axisIdx);
    } else {
      const std::optional<int64_t> extentAsConstant = tryConstantIndex(mappedExt);
      if (extentAsConstant && static_cast<int64_t>(ctx.allVectorSizes[axisIdx]) == *extentAsConstant)
        hits.push_back(axisIdx);
    }
  }
  if (hits.size() != 1) return std::nullopt;
  return hits.front();
}

static void reconcileValueDimOrderWithTileExtents(Value loadedVector, SmallVector<int> &sortedDims,
                                                  const LoopVectorizationCtx &ctx) {
  SmallVector<Value> perDimExtents;
  if (!gatherVectorDynExtents(loadedVector, perDimExtents)) return;
  if (perDimExtents.size() != sortedDims.size()) return;

  SmallVector<int> fromExtents;
  fromExtents.reserve(perDimExtents.size());
  llvm::SmallDenseSet<unsigned> usedAxes;
  for (Value ext : perDimExtents) {
    std::optional<unsigned> axisOpt = matchExtentValueToCtxAxis(ext, ctx);
    if (!axisOpt || !usedAxes.insert(*axisOpt).second) return;
    fromExtents.push_back(static_cast<int>(*axisOpt));
  }

  if (sortedDims.size() == fromExtents.size() && std::equal(sortedDims.begin(), sortedDims.end(), fromExtents.begin()))
    return;
  sortedDims = std::move(fromExtents);
}

static void collectVectorAxes(const LoopVectorizationCtx &ctx, SmallVectorImpl<std::pair<Value, int>> &axes) {
  axes.clear();
  DenseSet<Value> seen;
  for (auto &[op, dim] : ctx.allLoopToVectorDim) {
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp || !seen.insert(forOp.getInductionVar()).second) continue;
    axes.push_back({forOp.getInductionVar(), static_cast<int>(dim)});
  }
  if (ctx.vectorizationAxis && ctx.vectorAxisVectorDim && seen.insert(ctx.vectorizationAxis).second)
    axes.push_back({ctx.vectorizationAxis, static_cast<int>(*ctx.vectorAxisVectorDim)});
  llvm::sort(axes, [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
}

static bool affineExprCoeffForIndex(AffineExpr expr, ArrayRef<int64_t> operandCoeffs, unsigned numDims,
                                    int64_t &coeff) {
  coeff = 0;
  if (isa<AffineConstantExpr>(expr)) return true;
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    unsigned pos = dimExpr.getPosition();
    if (pos >= operandCoeffs.size()) return false;
    coeff = operandCoeffs[pos];
    return true;
  }
  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned pos = numDims + symbolExpr.getPosition();
    if (pos >= operandCoeffs.size()) return false;
    coeff = operandCoeffs[pos];
    return true;
  }

  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr) return false;

  int64_t lhsCoeff = 0;
  int64_t rhsCoeff = 0;
  if (!affineExprCoeffForIndex(binaryExpr.getLHS(), operandCoeffs, numDims, lhsCoeff) ||
      !affineExprCoeffForIndex(binaryExpr.getRHS(), operandCoeffs, numDims, rhsCoeff))
    return false;

  switch (binaryExpr.getKind()) {
    case AffineExprKind::Add:
      coeff = lhsCoeff + rhsCoeff;
      return true;
    case AffineExprKind::Mul:
      return lhsCoeff == 0 && rhsCoeff == 0;
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
      return lhsCoeff == 0 && rhsCoeff == 0;
    default:
      return false;
  }
}

static bool indexCoeffForVectorAxis(Value index, Value vectorAxis, int64_t &coeff) {
  coeff = 0;
  if (index == vectorAxis) {
    coeff = 1;
    return true;
  }

  Operation *definingOp = index.getDefiningOp();
  if (!definingOp) return true;

  if (auto addOp = dyn_cast<arith::AddIOp>(definingOp)) {
    int64_t lhsCoeff = 0;
    int64_t rhsCoeff = 0;
    if (!indexCoeffForVectorAxis(addOp.getLhs(), vectorAxis, lhsCoeff) ||
        !indexCoeffForVectorAxis(addOp.getRhs(), vectorAxis, rhsCoeff))
      return false;
    coeff = lhsCoeff + rhsCoeff;
    return true;
  }
  if (auto subOp = dyn_cast<arith::SubIOp>(definingOp)) {
    int64_t lhsCoeff = 0;
    int64_t rhsCoeff = 0;
    if (!indexCoeffForVectorAxis(subOp.getLhs(), vectorAxis, lhsCoeff) ||
        !indexCoeffForVectorAxis(subOp.getRhs(), vectorAxis, rhsCoeff))
      return false;
    coeff = lhsCoeff - rhsCoeff;
    return true;
  }
  if (auto affineApplyOp = dyn_cast<affine::AffineApplyOp>(definingOp)) {
    SmallVector<int64_t> operandCoeffs;
    operandCoeffs.reserve(affineApplyOp.getMapOperands().size());
    for (Value operand : affineApplyOp.getMapOperands()) {
      int64_t operandCoeff = 0;
      if (!indexCoeffForVectorAxis(operand, vectorAxis, operandCoeff)) return false;
      operandCoeffs.push_back(operandCoeff);
    }
    return affineExprCoeffForIndex(affineApplyOp.getAffineMap().getResult(0), operandCoeffs,
                                   affineApplyOp.getAffineMap().getNumDims(), coeff);
  }

  return !definitionGraphContainsValue(index, vectorAxis);
}

static int getVectorDimForIndex(Value index, const LoopVectorizationCtx &ctx) {
  int directDim = ctx.getVectorDimForIV(index);
  if (directDim >= 0) return directDim;

  SmallVector<std::pair<Value, int>> axes;
  collectVectorAxes(ctx, axes);
  int matchedDim = -1;
  for (auto [axis, dim] : axes) {
    int64_t coeff = 0;
    if (!indexCoeffForVectorAxis(index, axis, coeff)) return -1;
    if (coeff == 0) continue;
    if (coeff != 1 || matchedDim >= 0) return -1;
    matchedDim = dim;
  }
  return matchedDim;
}

static void collectLoadIndexVectorDims(memref::LoadOp loadOp, const LoopVectorizationCtx &ctx,
                                       SmallVectorImpl<int> &indexToDim, SmallVector<int> &activeInAppearOrder) {
  indexToDim.clear();
  activeInAppearOrder.clear();
  for (Value idx : loadOp.getIndices()) {
    int dim = getVectorDimForIndex(idx, ctx);
    indexToDim.push_back(dim);
    if (dim >= 0) activeInAppearOrder.push_back(dim);
  }
}

static bool loadIndicesAreLoopInvariant(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
  if (ctx.vf1FuncLevelNoAnchor) return false;
  if (!ctx.scalarLoop.getOperation()) return false;
  Value vecAxis = ctx.getVectorizationAxis();
  Value mappedVecAxis = ctx.valueMapping.lookupOrDefault(vecAxis);
  for (Value idx : loadOp.getIndices()) {
    if (getVectorDimForIndex(idx, ctx) >= 0) return false;
    if (idx == vecAxis || idx == mappedVecAxis) return false;
    if (idx.getParentBlock() == ctx.scalarLoop.getBody()) return false;
  }
  return true;
}

/// `collectLoadIndexVectorDims` produced no active axes: scalar indices vs tile, optionally VF1 transfer_read.
static Value vectorizeLoadSubRankZero(memref::LoadOp loadOp, LoopVectorizationCtx &ctx, Value mappedMemRef,
                                      Type elemType, Location loc) {
  SmallVector<Value> indices;
  indices.reserve(loadOp.getIndices().size());
  std::transform(loadOp.getIndices().begin(), loadOp.getIndices().end(), std::back_inserter(indices),
                 [&ctx](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });

  if (!ctx.vf1FuncLevelNoAnchor) {
    // Scalar SSA; arith/store broadcast to tile when mixed with !npuvector.
    return ctx.builder.create<memref::LoadOp>(loc, mappedMemRef, indices);
  }

  npuvector::NPUVectorType readVecType = npuvector::NPUVectorType::get({ctx.allVectorSizes[0]}, elemType);
  Value padding = ctx.builder.create<arith::ConstantOp>(loc, ctx.builder.getZeroAttr(elemType));
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[0]));
  dynamicSizes.push_back(
    ctx.allVectorSizeValues[0] ? ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[0])
                               : ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[0]));
  auto transferRead = ctx.builder.create<npuvector::TransferReadOp>(
    loc, readVecType, mappedMemRef, ValueRange(indices), padding, Value(),
    ValueRange(dynamicSizes), ValueRange(maxSizes));
  Value result = transferRead.getResult();
  SmallVector<int> axisPerResultDim = {0};
  reconcileValueDimOrderWithTileExtents(result, axisPerResultDim, ctx);
  ctx.valueDimOrder[result] = axisPerResultDim;
  return result;
}

static Value vectorizeLoad(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
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

    // Scalar load only; broadcast deferred to arith/store when mixed with !npuvector.
    return ctx.builder.create<memref::LoadOp>(loc, mappedMemRef, indices);
  }

  SmallVector<int> indexToDim;
  SmallVector<int> activeInAppearOrder;
  collectLoadIndexVectorDims(loadOp, ctx, indexToDim, activeInAppearOrder);

  const unsigned subRank = activeInAppearOrder.size();
  if (subRank == 0)
    return vectorizeLoadSubRankZero(loadOp, ctx, mappedMemRef, elemType, loc);

  SmallVector<int> sortedDims(activeInAppearOrder);
  llvm::sort(sortedDims);
  const bool analyzedTranspose = (subRank > 1) && (activeInAppearOrder != sortedDims);

  SmallVector<int64_t> readShapeWithDyn(subRank);
  for (unsigned k = 0; k < subRank; ++k) {
    int gid = activeInAppearOrder[k];
    Value dynVal = ctx.allVectorSizeValues[gid];
    readShapeWithDyn[k] = dynVal ? ShapedType::kDynamic : ctx.allVectorSizes[gid];
  }
  npuvector::NPUVectorType readVecType = npuvector::NPUVectorType::get(readShapeWithDyn, elemType);

  SmallVector<Value> indices;
  for (Value idx : loadOp.getIndices()) {
    Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
    indices.push_back(newIdx);
  }

  Value padding = ctx.builder.create<arith::ConstantOp>(loc, ctx.builder.getZeroAttr(elemType));

  // One extent + one max per vectorized axis (rank == subRank), including constants for
  // static tile axes so e.g. 512x?x64 passes three pairs, not only the `?` slot.
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  dynamicSizes.reserve(subRank);
  maxSizes.reserve(subRank);
  for (unsigned k = 0; k < subRank; ++k) {
    int gid = activeInAppearOrder[k];
    maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[gid]));
    if (ctx.allVectorSizeValues[gid])
      dynamicSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[gid]));
    else
      dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[gid]));
  }

  auto transferRead =
    ctx.builder.create<npuvector::TransferReadOp>(loc, readVecType, mappedMemRef, ValueRange(indices), padding, Value(),
                                                  ValueRange(dynamicSizes), ValueRange(maxSizes));

  Value result = transferRead.getResult();
  SmallVector<int64_t> transposePerm;
  if (analyzedTranspose) {
    transposePerm.resize(subRank);
    for (unsigned i = 0; i < subRank; ++i) {
      int targetGlobal = sortedDims[i];
      unsigned j = 0;
      for (; j < subRank; ++j) {
        if (activeInAppearOrder[j] == targetGlobal) break;
      }
      transposePerm[i] = static_cast<int64_t>(j);
    }
    result = ctx.builder.create<npuvector::TransposeOp>(loc, result, transposePerm);
  }

  // valueDimOrder[result][r] = global ctx axis id for **result** dimension r. transfer_read dim k
  // follows memref index order -> axis activeInAppearOrder[k]. npuvector.transpose: result dim r
  // comes from operand dim transposePerm[r] (see gatherVectorDynExtents on TransposeOp).
  SmallVector<int> axisPerResultDim(subRank);
  if (analyzedTranspose) {
    for (unsigned r = 0; r < subRank; ++r)
      axisPerResultDim[r] = activeInAppearOrder[static_cast<unsigned>(transposePerm[r])];
  } else {
    for (unsigned k = 0; k < subRank; ++k) axisPerResultDim[k] = activeInAppearOrder[k];
  }
  reconcileValueDimOrderWithTileExtents(result, axisPerResultDim, ctx);
  ctx.valueDimOrder[result] = axisPerResultDim;

  return result;
}

static LogicalResult buildStoreTargetTypeForDimOrder(memref::StoreOp storeOp, ArrayRef<int> storeDimOrder,
                                                     LoopVectorizationCtx &ctx, npuvector::NPUVectorType &outTy,
                                                     SmallVectorImpl<int> &outResultDimToCtxAxis) {
  const unsigned ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  Type elemType = storeOp.getMemRefType().getElementType();
  SmallVector<int64_t> brShape;
  brShape.reserve(storeDimOrder.size());
  for (int ax : storeDimOrder) {
    if (ax < 0 || static_cast<unsigned>(ax) >= ctxRank) return failure();
    const unsigned uax = static_cast<unsigned>(ax);
    brShape.push_back(ctx.allVectorSizeValues[uax] ? ShapedType::kDynamic : ctx.allVectorSizes[uax]);
  }
  outTy = npuvector::NPUVectorType::get(brShape, elemType);
  outResultDimToCtxAxis.assign(storeDimOrder.begin(), storeDimOrder.end());
  return success();
}

static bool ensureBroadcastStoreValue(memref::StoreOp storeOp, LoopVectorizationCtx &ctx,
                                      ArrayRef<int> storeDimOrder, Value storeValue, Value &vectorValue) {
  if (vectorValue && mlir::isa<npuvector::NPUVectorType>(vectorValue.getType())) return true;
  vectorValue = Value();
  Value valueToBcast = ctx.valueMapping.lookupOrDefault(storeValue);
  if (storeDimOrder.empty()) {
    vectorValue = vectorizeBroadcastScalar(valueToBcast, ctx);
  } else {
    npuvector::NPUVectorType targetTy;
    SmallVector<int> dimToCtx;
    if (failed(buildStoreTargetTypeForDimOrder(storeOp, storeDimOrder, ctx, targetTy, dimToCtx))) {
      storeOp.emitError("npuvector-vectorize: store index maps to invalid vector dim");
      return false;
    }
    vectorValue = vectorizeBroadcastScalar(valueToBcast, ctx, targetTy, dimToCtx);
  }
  if (!vectorValue) {
    storeOp.emitError("npuvector-vectorize: store value broadcast failed (see diagnostic on defining op / location)");
    return false;
  }
  ctx.valueMapping.map(storeValue, vectorValue);
  return true;
}

static bool validateMultiDimStoreValueAxes(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, Value vectorValue,
                                           int64_t vecRank, ArrayRef<int> storeDimOrder,
                                           SmallVector<int> &valueDimOrdOut) {
  const bool hadDimOrderForValue = ctx.valueDimOrder.count(vectorValue) != 0;
  if (hadDimOrderForValue) valueDimOrdOut = ctx.valueDimOrder[vectorValue];
  if (!hadDimOrderForValue || valueDimOrdOut.empty()) {
    storeOp.emitError(
      "npuvector-vectorize: multi-index / multi-rank store requires non-empty "
      "valueDimOrder on the stored vector value");
    return false;
  }
  if (static_cast<int64_t>(valueDimOrdOut.size()) != vecRank) {
    storeOp.emitError("npuvector-vectorize: valueDimOrder rank does not match stored npuvector rank");
    return false;
  }
  for (int axisIdx : valueDimOrdOut) {
    if (!llvm::is_contained(storeDimOrder, axisIdx)) {
      storeOp.emitError(
        "npuvector-vectorize: store valueDimOrder axis not present on store indices (cannot "
        "intersect)");
      return false;
    }
  }
  return true;
}

static bool intersectStoreAxesWithValueAxes(memref::StoreOp storeOp, ArrayRef<int> storeDimOrder,
                                            ArrayRef<int> valueDimOrd, SmallVector<int> &intersectOut) {
  llvm::DenseSet<int> valueAxisSet;
  for (int axisIdx : valueDimOrd) valueAxisSet.insert(axisIdx);
  llvm::DenseSet<int> seenIntersectAxis;
  intersectOut.clear();
  intersectOut.reserve(valueDimOrd.size());
  for (int axisIdx : storeDimOrder) {
    if (valueAxisSet.find(axisIdx) == valueAxisSet.end()) continue;
    if (!seenIntersectAxis.insert(axisIdx).second) continue;
    intersectOut.push_back(axisIdx);
  }
  if (intersectOut.size() != valueDimOrd.size()) {
    storeOp.emitError(
      "npuvector-vectorize: store-index vector axes (restricted to value axes) "
      "do not match vector rank (missing value axis on indices?)");
    return false;
  }
  return true;
}

static bool transposeStoreVectorIfAxesDiffer(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, Location loc,
                                             ArrayRef<int> valueDimOrd, ArrayRef<int> intersectStoreDimOrder,
                                             Value &vectorValue) {
  if (valueDimOrd == intersectStoreDimOrder) return true;
  SmallVector<int64_t> perm(intersectStoreDimOrder.size());
  for (unsigned rowIdx = 0; rowIdx < intersectStoreDimOrder.size(); ++rowIdx) {
    bool matched = false;
    for (unsigned j = 0; j < valueDimOrd.size(); ++j) {
      if (valueDimOrd[j] == intersectStoreDimOrder[rowIdx]) {
        perm[rowIdx] = static_cast<int64_t>(j);
        matched = true;
        break;
      }
    }
    if (!matched) {
      storeOp.emitError(
        "npuvector-vectorize: cannot map intersected store axis order to valueDimOrder for "
        "transpose");
      return false;
    }
  }
  vectorValue = ctx.builder.create<npuvector::TransposeOp>(loc, vectorValue, perm);
  ctx.valueDimOrder[vectorValue] = SmallVector<int>(intersectStoreDimOrder.begin(), intersectStoreDimOrder.end());
  return true;
}

static bool rankLiftStoreVectorIfExtraIndices(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, Location loc,
                                              ArrayRef<int> storeDimOrder, ArrayRef<int> intersectStoreDimOrder,
                                              Value &vectorValue) {
  if (storeDimOrder.size() <= intersectStoreDimOrder.size()) return true;
  npuvector::NPUVectorType targetTy;
  SmallVector<int> resultDimToCtxAxis;
  if (failed(buildStoreTargetTypeForDimOrder(storeOp, storeDimOrder, ctx, targetTy, resultDimToCtxAxis))) {
    storeOp.emitError("npuvector-vectorize: store index maps to invalid vector dim");
    return false;
  }
  Value lifted = vectorizeBroadcastScalar(vectorValue, ctx, targetTy, resultDimToCtxAxis);
  if (!lifted) {
    storeOp.emitError(
      "npuvector-vectorize: store rank-lift broadcast failed (see diagnostic "
      "on npuvector.broadcast / valueDimOrder)");
    return false;
  }
  vectorValue = lifted;
  return true;
}

static bool reorderStoreVectorForIndices(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, Location loc,
                                         ArrayRef<int> storeDimOrder, Value &vectorValue) {
  auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!npuVecType) return true;
  const int64_t vecRank = npuVecType.getRank();
  if (!(vecRank > 1 || storeDimOrder.size() > 1)) return true;

  SmallVector<int> valueDimOrd;
  if (!validateMultiDimStoreValueAxes(storeOp, ctx, vectorValue, vecRank, storeDimOrder, valueDimOrd))
    return false;
  SmallVector<int> intersectStoreDimOrder;
  if (!intersectStoreAxesWithValueAxes(storeOp, storeDimOrder, valueDimOrd, intersectStoreDimOrder))
    return false;
  if (!transposeStoreVectorIfAxesDiffer(storeOp, ctx, loc, valueDimOrd, intersectStoreDimOrder, vectorValue))
    return false;
  return rankLiftStoreVectorIfExtraIndices(storeOp, ctx, loc, storeDimOrder, intersectStoreDimOrder, vectorValue);
}

static void vectorizeStore(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  Location loc = storeOp.getLoc();

  SmallVector<int> storeDimOrder;
  for (Value idx : storeOp.getIndices()) {
    int dim = getVectorDimForIndex(idx, ctx);
    if (dim >= 0) storeDimOrder.push_back(dim);
  }

  Value storeValue = storeOp.getValue();
  Value vectorValue = ctx.valueMapping.lookupOrNull(storeValue);
  if (!ensureBroadcastStoreValue(storeOp, ctx, storeDimOrder, storeValue, vectorValue)) return;

  SmallVector<Value> indices;
  indices.reserve(storeOp.getIndices().size());
  std::transform(storeOp.getIndices().begin(), storeOp.getIndices().end(), std::back_inserter(indices),
                 [&ctx](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });

  if (!mlir::isa<npuvector::NPUVectorType>(vectorValue.getType()))
    llvm_unreachable("vectorizeStore: vector value must be NPUVectorType");

  if (!reorderStoreVectorForIndices(storeOp, ctx, loc, storeDimOrder, vectorValue)) return;

  Value memRef = storeOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);

  ctx.builder.create<npuvector::TransferWriteOp>(loc, Type(), vectorValue, mappedMemRef, ValueRange(indices), Value());
}

static void collectArithVecOperands(Operation *op, LoopVectorizationCtx &ctx, SmallVectorImpl<Value> &vecOperands,
                                    SmallVectorImpl<unsigned> &scalarIndices) {
  vecOperands.clear();
  scalarIndices.clear();
  for (auto [opIdx, operand] : llvm::enumerate(op->getOperands())) {
    Value mapped = ctx.valueMapping.lookupOrNull(operand);
    Value effectiveOperand = mapped ? mapped : operand;
    if (mlir::isa<npuvector::NPUVectorType>(effectiveOperand.getType())) {
      vecOperands.push_back(effectiveOperand);
      continue;
    }
    vecOperands.push_back(effectiveOperand);
    scalarIndices.push_back(opIdx);
  }
}

static bool pickHighestRankNpuVecType(const SmallVectorImpl<Value> &vecOperands,
                                      npuvector::NPUVectorType &outRefVecType) {
  int64_t bestRank = -1;
  for (Value vo : vecOperands) {
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vo.getType());
    if (!nvt) continue;
    int64_t r = static_cast<int64_t>(nvt.getRank());
    if (r > bestRank) {
      bestRank = r;
      outRefVecType = nvt;
    }
  }
  return bestRank >= 0;
}

static bool isScalarSlotOperand(unsigned opIdx, llvm::ArrayRef<unsigned> scalarIndices) {
  return llvm::is_contained(scalarIndices, opIdx);
}

static bool lookupPeerBroadcastAxes(Operation *arithOp, LoopVectorizationCtx &ctx,
                                    const SmallVectorImpl<Value> &vecOperands, llvm::ArrayRef<unsigned> scalarIndices,
                                    npuvector::NPUVectorType refVecType, SmallVectorImpl<int> &outAxes) {
  outAxes.clear();
  const unsigned refRank = static_cast<unsigned>(refVecType.getRank());
  for (unsigned j = 0, e = vecOperands.size(); j < e; ++j) {
    if (isScalarSlotOperand(j, scalarIndices)) continue;
    Value vo = vecOperands[j];
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vo.getType());
    if (!nvt || static_cast<unsigned>(nvt.getRank()) != refRank) continue;
    auto it = ctx.valueDimOrder.find(vo);
    if (it != ctx.valueDimOrder.end() && it->second.size() == refRank) {
      outAxes.assign(it->second.begin(), it->second.end());
      return true;
    }
  }
  emitError(arithOp->getLoc()) << "npuvector-vectorize: rank-lift broadcast needs a peer !npuvector at rank " << refRank
                               << " with non-empty valueDimOrder on this op's operands (no fallback)";
  return false;
}

static bool broadcastScalarSlotsToRef(SmallVectorImpl<Value> &vecOperands,
                                      const SmallVectorImpl<unsigned> &scalarIndices, bool haveRefVec,
                                      npuvector::NPUVectorType refVecType, llvm::ArrayRef<int> broadcastCtxAxes,
                                      LoopVectorizationCtx &ctx) {
  npuvector::NPUVectorType target = haveRefVec ? refVecType : npuvector::NPUVectorType{};
  for (unsigned idx : scalarIndices) {
    Value broadcasted = vectorizeBroadcastScalar(vecOperands[idx], ctx, target, broadcastCtxAxes);
    if (!broadcasted) return false;
    vecOperands[idx] = broadcasted;
  }
  return true;
}

static bool alignOperandRanksToRef(SmallVectorImpl<Value> &vecOperands, npuvector::NPUVectorType refVecType,
                                   llvm::ArrayRef<int> broadcastCtxAxes, LoopVectorizationCtx &ctx) {
  for (unsigned i = 0, e = vecOperands.size(); i != e; ++i) {
    auto cur = mlir::dyn_cast<npuvector::NPUVectorType>(vecOperands[i].getType());
    if (!cur) continue;
    auto want = npuvector::NPUVectorType::get(refVecType.getShape(), cur.getElementType());
    if (want == cur) continue;
    Value aligned = vectorizeBroadcastScalar(vecOperands[i], ctx, refVecType, broadcastCtxAxes);
    if (!aligned) return false;
    vecOperands[i] = aligned;
  }
  return true;
}

static bool arithUsesRenamedNpuvectorDialect(Operation *op) {
  return isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
             arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, arith::BitcastOp, arith::IndexCastOp,
             arith::IndexCastUIOp, arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op);
}

static Operation *createRenamedOrSameNameVecArith(Operation *op, Location loc, SmallVectorImpl<Value> &vecOperands,
                                                  SmallVectorImpl<Type> &vecResultTypes, LoopVectorizationCtx &ctx) {
  if (arithUsesRenamedNpuvectorDialect(op)) {
    StringRef arithOpName = op->getName().getStringRef();
    std::string npuvectorOpName = "npuvector." + arithOpName.split('.').second.str();
    OperationName npuvectorName(npuvectorOpName, ctx.builder.getContext());
    OperationState state(loc, npuvectorName);
    state.addOperands(vecOperands);
    state.addTypes(vecResultTypes);
    state.addAttributes(op->getAttrs());
    return ctx.builder.create(state);
  }
  return ctx.builder.create(loc, op->getName().getIdentifier(), vecOperands, vecResultTypes, op->getAttrs());
}

static void propagateValueDimOrderToFirstOperandWithOrder(LoopVectorizationCtx &ctx, ValueRange vecOperands,
                                                          Value result) {
  auto resultNvt = mlir::dyn_cast<npuvector::NPUVectorType>(result.getType());
  const unsigned expectRank = resultNvt ? static_cast<unsigned>(resultNvt.getRank()) : 0;
  for (Value vo : vecOperands) {
    if (!mlir::isa<npuvector::NPUVectorType>(vo.getType())) continue;
    auto ordIter = ctx.valueDimOrder.find(vo);
    if (ordIter == ctx.valueDimOrder.end()) continue;
    // Copy before operator[](result): inserting 'result' may rehash DenseMap and invalidate ordIter / references.
    SmallVector<int> orderCopy = ordIter->second;
    if (orderCopy.empty()) continue;
    if (expectRank != 0 && orderCopy.size() != expectRank) continue;
    ctx.valueDimOrder[result] = std::move(orderCopy);
    break;
  }
}

static void reportVectorizeArithOpFailure(Operation *op, llvm::StringRef reason, const LoopVectorizationCtx &ctx) {
  if (!ctx.vf1FuncLevelNoAnchor) return;
  emitError(op->getLoc()) << "npuvector-vectorize: vectorizeArithOp failed on '" << op->getName() << "': " << reason;
}

static Value vectorizeArithOp(Operation *op, LoopVectorizationCtx &ctx) {
  Location loc = op->getLoc();
  if (hasIndexResult(*op)) {
    reportVectorizeArithOpFailure(op, "index-typed result (use scalar clone path instead of vectorizeArithOp)", ctx);
    return nullptr;
  }
  mergeAncestorValueDimOrderMissingKeys(ctx);
  SmallVector<Value, 4> vecOperands;
  SmallVector<unsigned> scalarIndices;
  collectArithVecOperands(op, ctx, vecOperands, scalarIndices);

  npuvector::NPUVectorType refVecType;
  const bool haveRefVec = pickHighestRankNpuVecType(vecOperands, refVecType);
  SmallVector<int, 8> broadcastAxesBuf;
  llvm::ArrayRef<int> broadcastCtxAxes;
  if (haveRefVec) {
    if (!lookupPeerBroadcastAxes(op, ctx, vecOperands, scalarIndices, refVecType, broadcastAxesBuf)) {
      return nullptr;
    }
    broadcastCtxAxes = broadcastAxesBuf;
  }

  if (!broadcastScalarSlotsToRef(vecOperands, scalarIndices, haveRefVec, refVecType, broadcastCtxAxes, ctx)) {
    reportVectorizeArithOpFailure(op, "broadcastScalarSlotsToRef (vectorizeBroadcastScalar failed)", ctx);
    return nullptr;
  }
  if (haveRefVec && !alignOperandRanksToRef(vecOperands, refVecType, broadcastCtxAxes, ctx)) {
    reportVectorizeArithOpFailure(op, "alignOperandRanksToRef", ctx);
    return nullptr;
  }

  SmallVector<Type, 4> vecResultTypes;
  for (Value result : op->getResults()) {
    Type scalarType = result.getType();
    npuvector::NPUVectorType vecType =
      haveRefVec ? npuvector::NPUVectorType::get(refVecType.getShape(), scalarType) : ctx.getVectorType(scalarType);
    vecResultTypes.push_back(vecType);
  }

  Operation *vecOp = createRenamedOrSameNameVecArith(op, loc, vecOperands, vecResultTypes, ctx);
  if (!vecOp || vecOp->getNumResults() == 0) {
    reportVectorizeArithOpFailure(op, !vecOp ? "createRenamedOrSameNameVecArith returned null (unsupported op or "
                                               "invalid vector operands/result types for this named op)"
                                             : "createRenamedOrSameNameVecArith produced op with zero results",
                                  ctx);
    return nullptr;
  }

  Value vecResult = vecOp->getResult(0);
  propagateValueDimOrderToFirstOperandWithOrder(ctx, vecOperands, vecResult);
  return vecResult;
}

static bool resolveBroadcastSrcToDestAxes(Value sourceVal, const LoopVectorizationCtx &ctx,
                                          llvm::ArrayRef<int> resultDimToCtxAxis, int64_t dstRank,
                                          SmallVectorImpl<int64_t> &outAxes) {
  outAxes.clear();
  auto srcNvt = mlir::dyn_cast<npuvector::NPUVectorType>(sourceVal.getType());
  if (!srcNvt) return false;
  const int64_t srcRank = static_cast<int64_t>(srcNvt.getRank());
  if (srcRank <= 0 || srcRank >= dstRank) return false;

  auto orderIter = ctx.valueDimOrder.find(sourceVal);
  if (orderIter == ctx.valueDimOrder.end()) return false;
  const SmallVector<int> &srcDimToGlobalAxis = orderIter->second;
  if (static_cast<int64_t>(srcDimToGlobalAxis.size()) != srcRank) return false;

  const unsigned ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  if (dstRank > static_cast<int64_t>(ctxRank)) return false;
  const unsigned ctxOff = ctxRank - static_cast<unsigned>(dstRank);

  llvm::SmallDenseSet<int64_t> usedDestDim;
  for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
    const int globalCtxAxis = srcDimToGlobalAxis[static_cast<unsigned>(srcDim)];
    if (globalCtxAxis < 0 || static_cast<unsigned>(globalCtxAxis) >= ctxRank) return false;

    int64_t destDim = -1;
    if (!resultDimToCtxAxis.empty()) {
      if (static_cast<size_t>(dstRank) != resultDimToCtxAxis.size()) return false;
      for (int64_t resultDim = 0; resultDim < dstRank; ++resultDim) {
        if (resultDimToCtxAxis[static_cast<unsigned>(resultDim)] == globalCtxAxis) {
          destDim = resultDim;
          break;
        }
      }
    } else {
      const unsigned axisUnsigned = static_cast<unsigned>(globalCtxAxis);
      if (axisUnsigned < ctxOff) return false;
      const unsigned relative = axisUnsigned - ctxOff;
      if (relative >= static_cast<unsigned>(dstRank)) return false;
      destDim = static_cast<int64_t>(relative);
    }
    if (destDim < 0) return false;
    if (!usedDestDim.insert(destDim).second) return false;
    outAxes.push_back(destDim);
  }
  return static_cast<int64_t>(outAxes.size()) == srcRank;
}

static bool gatherBroadcastExtentOperands(Location loc, LoopVectorizationCtx &ctx, unsigned outRank,
                                          unsigned ctxRank, llvm::ArrayRef<int> resultDimToCtxAxis,
                                          SmallVectorImpl<Value> &dynamicSizes, SmallVectorImpl<Value> &maxSizes) {
  const unsigned ctxOff = ctxRank - outRank;
  dynamicSizes.reserve(outRank);
  maxSizes.reserve(outRank);
  for (unsigned i = 0; i < outRank; ++i) {
    unsigned axisIdx =
      resultDimToCtxAxis.empty() ? (ctxOff + i) : static_cast<unsigned>(resultDimToCtxAxis[i]);
    if (!resultDimToCtxAxis.empty() && axisIdx >= ctxRank) return false;
    maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[axisIdx]));
    if (ctx.allVectorSizeValues[axisIdx])
      dynamicSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[axisIdx]));
    else
      dynamicSizes.push_back(ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[axisIdx]));
  }
  return true;
}

static bool resolveRankLiftBroadcastAxes(Location loc, Value scalarVal, LoopVectorizationCtx &ctx,
                                         llvm::ArrayRef<int> resultDimToCtxAxis, int64_t srcRank, int64_t dstRank,
                                         SmallVectorImpl<int64_t> &dimAxes) {
  mergeAncestorValueDimOrderMissingKeys(ctx);
  if (resolveBroadcastSrcToDestAxes(scalarVal, ctx, resultDimToCtxAxis, dstRank, dimAxes)) return true;
  if (Operation *defOp = scalarVal.getDefiningOp()) {
    emitError(defOp->getLoc()) << "npuvector-vectorize: cannot infer npuvector.broadcast `dimension` for rank lift ("
                               << srcRank << "D -> " << dstRank
                               << "D): resolveBroadcastSrcToDestAxes failed; align source and peer "
                                  "valueDimOrder / resultDimToCtxAxis (broadcastCtxAxes)";
  } else {
    emitError(loc) << "npuvector-vectorize: cannot infer npuvector.broadcast `dimension` for rank lift (" << srcRank
                   << "D -> " << dstRank
                   << "D): resolveBroadcastSrcToDestAxes failed; align source and peer "
                      "valueDimOrder / resultDimToCtxAxis (broadcastCtxAxes)";
  }
  return false;
}

static Value vectorizeBroadcastScalar(Value scalarVal, LoopVectorizationCtx &ctx, npuvector::NPUVectorType targetType,
                                      llvm::ArrayRef<int> resultDimToCtxAxis) {
  Type elemType;
  if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(scalarVal.getType()))
    elemType = nvt.getElementType();
  else
    elemType = scalarVal.getType();

  if (mlir::isa<npuvector::NPUVectorType>(scalarVal.getType())) {
    if (targetType) {
      auto want = npuvector::NPUVectorType::get(targetType.getShape(), elemType);
      if (want == scalarVal.getType()) return scalarVal;
    } else {
      return scalarVal;
    }
  }

  Location loc = scalarVal.getLoc();

  npuvector::NPUVectorType vecType =
    targetType ? npuvector::NPUVectorType::get(targetType.getShape(), elemType) : ctx.getVectorType(elemType);

  const unsigned outRank = static_cast<unsigned>(vecType.getRank());
  const unsigned ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  if (outRank > ctxRank) return Value();
  if (!resultDimToCtxAxis.empty() && resultDimToCtxAxis.size() != outRank) return Value();

  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  if (!gatherBroadcastExtentOperands(loc, ctx, outRank, ctxRank, resultDimToCtxAxis, dynamicSizes, maxSizes))
    return Value();

  const int64_t dstRank = static_cast<int64_t>(vecType.getRank());
  int64_t srcRank = 0;
  if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(scalarVal.getType()))
    srcRank = static_cast<int64_t>(nvt.getRank());
  SmallVector<int64_t> dimAxes;
  if (srcRank > 0 && srcRank < dstRank &&
      !resolveRankLiftBroadcastAxes(loc, scalarVal, ctx, resultDimToCtxAxis, srcRank, dstRank, dimAxes))
    return Value();

  DenseI64ArrayAttr dimAttr = ctx.builder.getDenseI64ArrayAttr(dimAxes);
  auto broadcast = ctx.builder.create<npuvector::BroadcastOp>(loc, vecType, scalarVal, ValueRange(dynamicSizes),
                                                              ValueRange(maxSizes), dimAttr);

  if (!resultDimToCtxAxis.empty() && resultDimToCtxAxis.size() == outRank) {
    ctx.valueDimOrder[broadcast.getResult()] = SmallVector<int>(resultDimToCtxAxis.begin(), resultDimToCtxAxis.end());
  }
  return broadcast.getResult();
}

static void cloneScalarOp(Operation &op, LoopVectorizationCtx &ctx) {
  OpBuilder::InsertionGuard guard(ctx.builder);

  IRMapping mapper;
  for (const auto &kv : ctx.valueMapping.getValueMap()) mapper.map(kv.first, kv.second);

  Operation *clonedOp = ctx.builder.clone(op, mapper);

  clonedOp->removeAttr(kSkipVectorizeAttr);

  for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
    Value mappedValue = ctx.valueMapping.lookupOrDefault(operand);
    clonedOp->setOperand(idx, mappedValue);
  }

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i)
    ctx.valueMapping.map(op.getResult(i), clonedOp->getResult(i));
}

static LogicalResult vectorizeRegion(Region &region, LoopVectorizationCtx &ctx);

enum class SignedEuclideanDivKind { FloorTowardNegInfinity, CeilTowardPosInfinity };

static int64_t signedDivEuclidean(int64_t numerator, int64_t denominator, SignedEuclideanDivKind kind) {
  assert(denominator != 0 && "signedDivEuclidean: zero denominator");
  const int64_t quotient = numerator / denominator;
  const int64_t remainder = numerator % denominator;
  if (kind == SignedEuclideanDivKind::FloorTowardNegInfinity) {
    if (remainder != 0 && ((remainder > 0) != (denominator > 0))) return quotient - 1;
  } else if (remainder != 0 && ((remainder > 0) == (denominator > 0))) {
    return quotient + 1;
  }
  return quotient;
}

struct CompareSliceBounds {
  bool recognized;
  bool predicateEmptyOnIntegers;
  bool hasFiniteLowerInclusive;
  int64_t lowerInclusive;
  bool hasFiniteUpperExclusive;
  int64_t upperExclusive;
};

static void markCompareSliceEmptyOnIntegers(CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = true;
  result.hasFiniteLowerInclusive = false;
  result.hasFiniteUpperExclusive = false;
}

static void markCompareSliceFullLineUnconstrained(CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  result.hasFiniteLowerInclusive = false;
  result.hasFiniteUpperExclusive = false;
}

static void clampNonnegativeBounds(CompareSliceBounds &bounds) {
  if (!bounds.recognized || bounds.predicateEmptyOnIntegers) return;
  if (bounds.hasFiniteLowerInclusive && bounds.lowerInclusive < 0) bounds.lowerInclusive = 0;
  if (bounds.hasFiniteLowerInclusive && bounds.hasFiniteUpperExclusive &&
      bounds.lowerInclusive >= bounds.upperExclusive) {
    bounds.predicateEmptyOnIntegers = true;
    bounds.hasFiniteLowerInclusive = false;
    bounds.hasFiniteUpperExclusive = false;
  }
}

static void fillEqSlice(int64_t a, int64_t K, CompareSliceBounds &result) {
  const int64_t remainder = K % a;
  if (remainder != 0) {
    markCompareSliceEmptyOnIntegers(result);
    return;
  }
  const int64_t ivSolution = K / a;
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  result.hasFiniteLowerInclusive = true;
  result.lowerInclusive = ivSolution;
  result.hasFiniteUpperExclusive = true;
  result.upperExclusive = ivSolution + 1;
  clampNonnegativeBounds(result);
}

static void fillSgeSlice(int64_t a, int64_t K, CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  if (a > 0) {
    result.hasFiniteLowerInclusive = true;
    result.lowerInclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::CeilTowardPosInfinity);
    result.hasFiniteUpperExclusive = false;
  } else {
    result.hasFiniteLowerInclusive = false;
    result.hasFiniteUpperExclusive = true;
    result.upperExclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::FloorTowardNegInfinity) + 1;
  }
  clampNonnegativeBounds(result);
}

static void fillSleSlice(int64_t a, int64_t K, CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  if (a > 0) {
    result.hasFiniteLowerInclusive = false;
    result.hasFiniteUpperExclusive = true;
    result.upperExclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::FloorTowardNegInfinity) + 1;
  } else {
    result.hasFiniteLowerInclusive = true;
    result.lowerInclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::CeilTowardPosInfinity);
    result.hasFiniteUpperExclusive = false;
  }
  clampNonnegativeBounds(result);
}

static void fillSgtSlice(int64_t a, int64_t K, int64_t kMin, int64_t kMax, CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  if (a > 0) {
    if (K == kMax) {
      markCompareSliceEmptyOnIntegers(result);
      return;
    }
    result.hasFiniteLowerInclusive = true;
    result.lowerInclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::FloorTowardNegInfinity) + 1;
    result.hasFiniteUpperExclusive = false;
  } else {
    if (K == kMin) {
      markCompareSliceEmptyOnIntegers(result);
      return;
    }
    result.hasFiniteLowerInclusive = false;
    result.hasFiniteUpperExclusive = true;
    result.upperExclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::CeilTowardPosInfinity);
  }
  clampNonnegativeBounds(result);
}

static void fillSltSlice(int64_t a, int64_t K, int64_t kMin, int64_t kMax, CompareSliceBounds &result) {
  result.recognized = true;
  result.predicateEmptyOnIntegers = false;
  if (a > 0) {
    if (K == kMin) {
      markCompareSliceEmptyOnIntegers(result);
      return;
    }
    result.hasFiniteLowerInclusive = false;
    result.hasFiniteUpperExclusive = true;
    result.upperExclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::CeilTowardPosInfinity);
  } else {
    if (K == kMax) {
      markCompareSliceEmptyOnIntegers(result);
      return;
    }
    result.hasFiniteLowerInclusive = true;
    result.lowerInclusive = signedDivEuclidean(K, a, SignedEuclideanDivKind::FloorTowardNegInfinity) + 1;
    result.hasFiniteUpperExclusive = false;
  }
  clampNonnegativeBounds(result);
}

static void dispatchNonzeroCompare(arith::CmpIPredicate effectivePredicate, int64_t a, int64_t K, int64_t kMin,
                                   int64_t kMax, CompareSliceBounds &result) {
  switch (effectivePredicate) {
    case arith::CmpIPredicate::eq:
      fillEqSlice(a, K, result);
      return;
    case arith::CmpIPredicate::sge:
      fillSgeSlice(a, K, result);
      return;
    case arith::CmpIPredicate::sle:
      fillSleSlice(a, K, result);
      return;
    case arith::CmpIPredicate::sgt:
      fillSgtSlice(a, K, kMin, kMax, result);
      return;
    case arith::CmpIPredicate::slt:
      fillSltSlice(a, K, kMin, kMax, result);
      return;
    default:
      result.recognized = false;
      return;
  }
}

static CompareSliceBounds fillZeroCoefficientCompare(arith::CmpIPredicate effectivePredicate, int64_t b, int64_t R) {
  CompareSliceBounds result{};
  result.recognized = false;
  result.predicateEmptyOnIntegers = false;
  result.hasFiniteLowerInclusive = false;
  result.hasFiniteUpperExclusive = false;

  const int64_t lhsConstant = b;
  switch (effectivePredicate) {
    case arith::CmpIPredicate::eq:
      if (lhsConstant == R)
        markCompareSliceFullLineUnconstrained(result);
      else
        markCompareSliceEmptyOnIntegers(result);
      return result;
    case arith::CmpIPredicate::slt:
      if (lhsConstant < R)
        markCompareSliceFullLineUnconstrained(result);
      else
        markCompareSliceEmptyOnIntegers(result);
      return result;
    case arith::CmpIPredicate::sle:
      if (lhsConstant <= R)
        markCompareSliceFullLineUnconstrained(result);
      else
        markCompareSliceEmptyOnIntegers(result);
      return result;
    case arith::CmpIPredicate::sgt:
      if (lhsConstant > R)
        markCompareSliceFullLineUnconstrained(result);
      else
        markCompareSliceEmptyOnIntegers(result);
      return result;
    case arith::CmpIPredicate::sge:
      if (lhsConstant >= R)
        markCompareSliceFullLineUnconstrained(result);
      else
        markCompareSliceEmptyOnIntegers(result);
      return result;
    default:
      return result;
  }
}

static CompareSliceBounds deriveLinearCompareBounds(arith::CmpIPredicate originalPredicate,
                                                    int64_t linearCoefficientOfInductionVar,
                                                    int64_t affineConstantOffset, int64_t rightHandSideIndexConstant) {
  CompareSliceBounds result{};
  result.recognized = false;
  result.predicateEmptyOnIntegers = false;
  result.hasFiniteLowerInclusive = false;
  result.hasFiniteUpperExclusive = false;

  if (originalPredicate == arith::CmpIPredicate::ne) return result;

  arith::CmpIPredicate effectivePredicate = originalPredicate;
  const int64_t a = linearCoefficientOfInductionVar;
  const int64_t b = affineConstantOffset;
  const int64_t R = rightHandSideIndexConstant;

  if (originalPredicate == arith::CmpIPredicate::ult || originalPredicate == arith::CmpIPredicate::ule ||
      originalPredicate == arith::CmpIPredicate::ugt || originalPredicate == arith::CmpIPredicate::uge) {
    if (a < 0 || b < 0 || R < 0) return result;
    switch (originalPredicate) {
      case arith::CmpIPredicate::uge:
        effectivePredicate = arith::CmpIPredicate::sge;
        break;
      case arith::CmpIPredicate::ugt:
        effectivePredicate = arith::CmpIPredicate::sgt;
        break;
      case arith::CmpIPredicate::ule:
        effectivePredicate = arith::CmpIPredicate::sle;
        break;
      case arith::CmpIPredicate::ult:
        effectivePredicate = arith::CmpIPredicate::slt;
        break;
      default:
        return result;
    }
  }

  const int64_t K = R - b;
  const int64_t kMin = std::numeric_limits<int64_t>::min();
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  if (a == -1 && (K == kMin || K == kMax)) return result;

  if (a == 0) return fillZeroCoefficientCompare(effectivePredicate, b, R);

  dispatchNonzeroCompare(effectivePredicate, a, K, kMin, kMax, result);
  return result;
}

static bool decomposeAffineOneDimension(AffineExpr affineExpr, int64_t &coefficientOfDim0, int64_t &constantOffset);

static bool decomposeAffineBinaryChildren(AffineExpr lhs, AffineExpr rhs, int64_t &leftCoeff, int64_t &leftConst,
                                          int64_t &rightCoeff, int64_t &rightConst) {
  return decomposeAffineOneDimension(lhs, leftCoeff, leftConst) &&
         decomposeAffineOneDimension(rhs, rightCoeff, rightConst);
}

static bool decomposeAffineOneDimension(AffineExpr affineExpr, int64_t &coefficientOfDim0, int64_t &constantOffset) {
  coefficientOfDim0 = 0;
  constantOffset = 0;
  if (auto constantExpr = dyn_cast<AffineConstantExpr>(affineExpr)) {
    constantOffset = constantExpr.getValue();
    return true;
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(affineExpr)) {
    if (dimExpr.getPosition() != 0) return false;
    coefficientOfDim0 = 1;
    return true;
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(affineExpr);
  if (!binaryExpr) return false;
  if (binaryExpr.getKind() == AffineExprKind::Add) {
    int64_t leftCoeff = 0, leftConst = 0, rightCoeff = 0, rightConst = 0;
    if (!decomposeAffineBinaryChildren(binaryExpr.getLHS(), binaryExpr.getRHS(), leftCoeff, leftConst, rightCoeff,
                                       rightConst))
      return false;
    coefficientOfDim0 = leftCoeff + rightCoeff;
    constantOffset = leftConst + rightConst;
    return true;
  }
  if (binaryExpr.getKind() == AffineExprKind::Mul) {
    int64_t leftCoeff = 0, leftConst = 0, rightCoeff = 0, rightConst = 0;
    if (!decomposeAffineBinaryChildren(binaryExpr.getLHS(), binaryExpr.getRHS(), leftCoeff, leftConst, rightCoeff,
                                       rightConst))
      return false;
    if (leftCoeff != 0 && rightCoeff != 0) return false;
    coefficientOfDim0 = leftCoeff * rightConst + rightCoeff * leftConst;
    constantOffset = leftConst * rightConst;
    return true;
  }
  return false;
}

static bool definitionGraphContainsValue(Value conditionSsaValue, Value targetSsaValue) {
  SmallVector<Value, 8> worklist;
  DenseSet<Value> visited;
  worklist.push_back(conditionSsaValue);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == targetSsaValue) return true;
    if (!visited.insert(current).second) continue;
    Operation *definingOp = current.getDefiningOp();
    if (!definingOp) continue;
    worklist.insert(worklist.end(), definingOp->operand_begin(), definingOp->operand_end());
  }
  return false;
}

static LogicalResult vectorizeRegionBodyOnly(Region &region, LoopVectorizationCtx &ctx) {
  Block *block = &region.front();
  for (Operation &op : block->without_terminator()) {
    if (failed(vectorizeOneOp(op, ctx))) return failure();
  }
  return success();
}

static void emitVectorizedStore(memref::StoreOp storeOp, Value vecVal, LoopVectorizationCtx &ctx) {
  Location loc = storeOp.getLoc();
  SmallVector<Value> indices;
  ValueRange rawIndices = storeOp.getIndices();
  indices.reserve(rawIndices.size());
  std::transform(rawIndices.begin(), rawIndices.end(), std::back_inserter(indices),
                 [&](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(storeOp.getMemRef());
  ctx.builder.create<npuvector::TransferWriteOp>(loc, Type(), vecVal, mappedMemRef, ValueRange(indices), Value());
}

static bool emitIfSlice(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc, Value inductionVar,
                        CompareSliceBounds sliceBounds) {
  OpBuilder &opBuilder = ctx.builder;
  int sliceAxisDimI = ctx.getVectorDimForIV(inductionVar);
  unsigned dimForTileOnSliceAxis;
  if (sliceAxisDimI >= 0)
    dimForTileOnSliceAxis = static_cast<unsigned>(sliceAxisDimI);
  else if (!ctx.allVectorSizeValues.empty())
    dimForTileOnSliceAxis = static_cast<unsigned>(ctx.allVectorSizeValues.size() - 1U);
  else
    return false;
  if (dimForTileOnSliceAxis >= ctx.allVectorSizeValues.size() || dimForTileOnSliceAxis >= ctx.allVectorSizes.size())
    return false;

  Value tileLowerBoundOnVectorAxis = ctx.valueMapping.lookupOrDefault(inductionVar);
  Value vectorExtentAlongAxisForThisTile;
  if (ctx.allVectorSizeValues[dimForTileOnSliceAxis]) {
    vectorExtentAlongAxisForThisTile = ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[dimForTileOnSliceAxis]);
  } else {
    vectorExtentAlongAxisForThisTile =
      opBuilder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[dimForTileOnSliceAxis]);
  }
  Value tileHalfOpenUpperBound =
    opBuilder.create<arith::AddIOp>(loc, tileLowerBoundOnVectorAxis, vectorExtentAlongAxisForThisTile);

  Value sliceLowerBoundOnVectorAxis = tileLowerBoundOnVectorAxis;
  if (sliceBounds.hasFiniteLowerInclusive) {
    Value constantPredicateLowerInclusive = opBuilder.create<arith::ConstantIndexOp>(loc, sliceBounds.lowerInclusive);
    sliceLowerBoundOnVectorAxis =
      opBuilder.create<arith::MaxSIOp>(loc, tileLowerBoundOnVectorAxis, constantPredicateLowerInclusive);
  }

  Value sliceHalfOpenUpperBound = tileHalfOpenUpperBound;
  if (sliceBounds.hasFiniteUpperExclusive) {
    Value constantPredicateUpperExclusive = opBuilder.create<arith::ConstantIndexOp>(loc, sliceBounds.upperExclusive);
    sliceHalfOpenUpperBound =
      opBuilder.create<arith::MinSIOp>(loc, tileHalfOpenUpperBound, constantPredicateUpperExclusive);
  }

  Value sliceHalfOpenUpperClampedNotBelowSliceLower =
    opBuilder.create<arith::MaxSIOp>(loc, sliceHalfOpenUpperBound, sliceLowerBoundOnVectorAxis);
  Value vectorLengthForThenRegionAlongAxis =
    opBuilder.create<arith::SubIOp>(loc, sliceHalfOpenUpperClampedNotBelowSliceLower, sliceLowerBoundOnVectorAxis);

  if (sliceBounds.predicateEmptyOnIntegers) {
    vectorLengthForThenRegionAlongAxis = opBuilder.create<arith::ConstantIndexOp>(loc, 0);
    sliceLowerBoundOnVectorAxis = tileLowerBoundOnVectorAxis;
  }

  Value constantIndexZero = opBuilder.create<arith::ConstantIndexOp>(loc, 0);
  Value thenRegionGuardNonEmptyExtent = opBuilder.create<arith::CmpIOp>(
    loc, arith::CmpIPredicate::ne, vectorLengthForThenRegionAlongAxis, constantIndexZero);

  auto vectorizedIfOp = opBuilder.create<scf::IfOp>(loc, TypeRange{}, thenRegionGuardNonEmptyExtent, false);
  {
    OpBuilder::InsertionGuard insertionGuard(opBuilder);
    opBuilder.setInsertionPointToStart(vectorizedIfOp.thenBlock());
    LoopVectorizationCtx thenRegionLoopVectorizationCtx = ctx;
    thenRegionLoopVectorizationCtx.valueMapping.map(inductionVar, sliceLowerBoundOnVectorAxis);
    thenRegionLoopVectorizationCtx.allVectorSizeValues[dimForTileOnSliceAxis] = vectorLengthForThenRegionAlongAxis;
    if (failed(vectorizeRegion(ifOp.getThenRegion(), thenRegionLoopVectorizationCtx))) {
      vectorizedIfOp.erase();
      return false;
    }
  }
  opBuilder.setInsertionPointAfter(vectorizedIfOp);
  return true;
}

static bool emitIfSliceWithElse(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc, Value inductionVar,
                                CompareSliceBounds sliceBounds, memref::StoreOp consumerStore) {
  OpBuilder &b = ctx.builder;
  int axisDimI = ctx.getVectorDimForIV(inductionVar);
  unsigned dTile;
  if (axisDimI >= 0)
    dTile = static_cast<unsigned>(axisDimI);
  else if (!ctx.allVectorSizeValues.empty())
    dTile = static_cast<unsigned>(ctx.allVectorSizeValues.size() - 1U);
  else
    return false;
  if (dTile >= ctx.allVectorSizeValues.size() || dTile >= ctx.allVectorSizes.size()) return false;

  Value tileLb = ctx.valueMapping.lookupOrDefault(inductionVar);
  Value tileVF;
  if (ctx.allVectorSizeValues[dTile])
    tileVF = ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[dTile]);
  else
    tileVF = b.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[dTile]);
  Value tileUb = b.create<arith::AddIOp>(loc, tileLb, tileVF);

  Value thenSliceLb = tileLb;
  if (sliceBounds.hasFiniteLowerInclusive)
    thenSliceLb =
      b.create<arith::MaxSIOp>(loc, tileLb, b.create<arith::ConstantIndexOp>(loc, sliceBounds.lowerInclusive));
  Value thenSliceUb = tileUb;
  if (sliceBounds.hasFiniteUpperExclusive)
    thenSliceUb =
      b.create<arith::MinSIOp>(loc, tileUb, b.create<arith::ConstantIndexOp>(loc, sliceBounds.upperExclusive));
  thenSliceUb = b.create<arith::MaxSIOp>(loc, thenSliceUb, thenSliceLb);
  Value thenLen = b.create<arith::SubIOp>(loc, thenSliceUb, thenSliceLb);

  if (sliceBounds.predicateEmptyOnIntegers) {
    thenLen = b.create<arith::ConstantIndexOp>(loc, 0);
    thenSliceLb = tileLb;
  }

  Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);

  unsigned axisDim = dTile;

  // --- then slice ---
  Value thenGuard = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, thenLen, c0);
  auto thenIfOp = b.create<scf::IfOp>(loc, TypeRange{}, thenGuard, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(thenIfOp.thenBlock());
    LoopVectorizationCtx thenCtx = ctx;
    thenCtx.valueMapping.map(inductionVar, thenSliceLb);
    thenCtx.allVectorSizeValues[axisDim] = thenLen;
    if (failed(vectorizeRegionBodyOnly(ifOp.getThenRegion(), thenCtx))) {
      thenIfOp.erase();
      return false;
    }
    auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    Value yieldScalar = thenYield.getOperand(0);
    Value yieldVec = thenCtx.valueMapping.lookupOrNull(yieldScalar);
    if (!yieldVec) {
      yieldVec = vectorizeBroadcastScalar(yieldScalar, thenCtx);
      if (!yieldVec) {
        thenIfOp.erase();
        return false;
      }
    }
    emitVectorizedStore(consumerStore, yieldVec, thenCtx);
  }
  b.setInsertionPointAfter(thenIfOp);

  // --- else slice ---
  Value elseSliceLb = thenSliceUb;
  Value elseLen = b.create<arith::SubIOp>(loc, tileUb, elseSliceLb);
  elseLen = b.create<arith::MaxSIOp>(loc, elseLen, c0);
  Value elseGuard = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, elseLen, c0);
  auto elseIfOp = b.create<scf::IfOp>(loc, TypeRange{}, elseGuard, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(elseIfOp.thenBlock());
    LoopVectorizationCtx elseCtx = ctx;
    elseCtx.valueMapping.map(inductionVar, elseSliceLb);
    elseCtx.allVectorSizeValues[axisDim] = elseLen;
    auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    Value elseScalar = elseYield.getOperand(0);
    Value elseVec = vectorizeBroadcastScalar(elseScalar, elseCtx);
    if (!elseVec) {
      elseIfOp.erase();
      return false;
    }
    emitVectorizedStore(consumerStore, elseVec, elseCtx);
  }
  b.setInsertionPointAfter(elseIfOp);

  ctx.absorbedOps.insert(consumerStore.getOperation());
  return true;
}

static bool tryRecognizeIVDependentSlice(Value condition, Value inductionVar, CompareSliceBounds &sliceBounds) {
  auto compareIntegerOp = condition.getDefiningOp<arith::CmpIOp>();
  if (!compareIntegerOp) return false;
  auto rhsConst = tryConstantIndex(compareIntegerOp.getRhs());
  if (!rhsConst) return false;
  auto affineApplyOp = compareIntegerOp.getLhs().getDefiningOp<affine::AffineApplyOp>();
  if (!affineApplyOp) return false;
  AffineMap affineMap = affineApplyOp.getAffineMap();
  if (affineMap.getNumDims() != 1 || affineMap.getNumSymbols() != 0 || affineMap.getNumResults() != 1 ||
      affineApplyOp.getDimOperands().size() != 1 || affineApplyOp.getDimOperands()[0] != inductionVar)
    return false;
  int64_t coeff = 0, offset = 0;
  if (!decomposeAffineOneDimension(affineMap.getResult(0), coeff, offset)) return false;
  sliceBounds = deriveLinearCompareBounds(compareIntegerOp.getPredicate(), coeff, offset, *rhsConst);
  return sliceBounds.recognized;
}

static memref::StoreOp findUniqueStoreConsumer(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 1) return nullptr;
  Value result = ifOp.getResult(0);
  memref::StoreOp consumer = nullptr;
  for (Operation *user : result.getUsers()) {
    auto store = dyn_cast<memref::StoreOp>(user);
    if (!store || consumer) return nullptr;
    consumer = store;
  }
  return consumer;
}

static bool emitIfWithElseAndStoreNoSlice(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc, Value cond,
                                          memref::StoreOp consumerStore) {
  OpBuilder &b = ctx.builder;
  auto outerIf = b.create<scf::IfOp>(loc, TypeRange{}, cond, true);
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outerIf.thenBlock());
    if (failed(vectorizeRegionBodyOnly(ifOp.getThenRegion(), ctx))) {
      outerIf.erase();
      return false;
    }
    auto thenY = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    Value yieldVec = ctx.valueMapping.lookupOrNull(thenY.getOperand(0));
    if (!yieldVec) {
      yieldVec = vectorizeBroadcastScalar(thenY.getOperand(0), ctx);
      if (!yieldVec) {
        outerIf.erase();
        return false;
      }
    }
    emitVectorizedStore(consumerStore, yieldVec, ctx);
  }
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(outerIf.elseBlock());
    if (failed(vectorizeRegionBodyOnly(ifOp.getElseRegion(), ctx))) {
      outerIf.erase();
      return false;
    }
    auto elseY = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
    Value elseVec = ctx.valueMapping.lookupOrNull(elseY.getOperand(0));
    if (!elseVec) {
      elseVec = vectorizeBroadcastScalar(elseY.getOperand(0), ctx);
      if (!elseVec) {
        outerIf.erase();
        return false;
      }
    }
    emitVectorizedStore(consumerStore, elseVec, ctx);
  }
  b.setInsertionPointAfter(outerIf);
  ctx.absorbedOps.insert(consumerStore.getOperation());
  return true;
}

static Value resolveIfDependentIV(Value condition, LoopVectorizationCtx &ctx) {
  Value currentIV = ctx.scalarLoop.getInductionVar();
  if (hasVectorizationAttr(ctx.scalarLoop.getOperation()) &&
      definitionGraphContainsValue(condition, currentIV))
    return currentIV;
  for (auto &[loopOp, dimIgnored] : ctx.allLoopToVectorDim) {
    (void)dimIgnored;
    auto ancestorFor = dyn_cast<scf::ForOp>(loopOp);
    if (!ancestorFor || ancestorFor == ctx.scalarLoop) continue;
    if (definitionGraphContainsValue(condition, ancestorFor.getInductionVar()))
      return ancestorFor.getInductionVar();
  }
  return Value();
}

static Value vectorizeIfElseWithVectorResults(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc,
                                              Value vecCondition) {
  SmallVector<Type> vecResultTypes;
  TypeRange resultTypes = ifOp.getResultTypes();
  vecResultTypes.reserve(resultTypes.size());
  std::transform(resultTypes.begin(), resultTypes.end(), std::back_inserter(vecResultTypes),
                 [&](Type elemTy) { return ctx.getVectorType(elemTy); });
  auto vecIfOp = ctx.builder.create<scf::IfOp>(loc, vecResultTypes, vecCondition, true);
  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(vecIfOp.thenBlock());
    if (failed(vectorizeRegion(ifOp.getThenRegion(), ctx))) {
      vecIfOp.erase();
      return nullptr;
    }
  }
  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(vecIfOp.elseBlock());
    if (failed(vectorizeRegion(ifOp.getElseRegion(), ctx))) {
      vecIfOp.erase();
      return nullptr;
    }
  }
  ctx.builder.setInsertionPointAfter(vecIfOp);
  return vecIfOp.getResult(0);
}

enum class IvDependentIfRewrite { None, Failed, Finished };

static IvDependentIfRewrite rewriteIvDependentIfSlices(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc,
                                                       Value dependentIV, Value condition, bool hasElse,
                                                       bool hasResults) {
  CompareSliceBounds sliceBounds{};
  if (!tryRecognizeIVDependentSlice(condition, dependentIV, sliceBounds)) return IvDependentIfRewrite::None;
  if (hasElse && hasResults) {
    memref::StoreOp consumer = findUniqueStoreConsumer(ifOp);
    if (!consumer) return IvDependentIfRewrite::Failed;
    if (!emitIfSliceWithElse(ifOp, ctx, loc, dependentIV, sliceBounds, consumer))
      return IvDependentIfRewrite::Failed;
    return IvDependentIfRewrite::Finished;
  }
  if (!hasElse && !hasResults) {
    if (!emitIfSlice(ifOp, ctx, loc, dependentIV, sliceBounds)) return IvDependentIfRewrite::Failed;
    return IvDependentIfRewrite::Finished;
  }
  return IvDependentIfRewrite::None;
}

static Value vectorizeIf(scf::IfOp ifOp, LoopVectorizationCtx &ctx) {
  Location loc = ifOp.getLoc();
  Value condition = ifOp.getCondition();
  bool hasElse = !ifOp.getElseRegion().empty();
  bool hasResults = ifOp.getNumResults() > 0;

  Value dependentIV = resolveIfDependentIV(condition, ctx);

  if (dependentIV) {
    IvDependentIfRewrite ivRewrite =
      rewriteIvDependentIfSlices(ifOp, ctx, loc, dependentIV, condition, hasElse, hasResults);
    if (ivRewrite == IvDependentIfRewrite::Failed) return nullptr;
    if (ivRewrite == IvDependentIfRewrite::Finished) return Value();
  }

  // Non-IV-dependent + else + results: prefer `transfer_write` absorption (no npuvector if
  // result) when the if result has a single memref.store user; otherwise keep a vectorized
  // `scf.if` that yields npuvector (e.g. tail `if` feeding `npuvector.broadcast` on the result).
  if (hasElse && hasResults && !dependentIV) {
    Value vecCondition = ctx.valueMapping.lookupOrNull(condition);
    if (!vecCondition) vecCondition = condition;
    if (memref::StoreOp consumer = findUniqueStoreConsumer(ifOp)) {
      if (emitIfWithElseAndStoreNoSlice(ifOp, ctx, loc, vecCondition, consumer)) return Value();
    }
    return vectorizeIfElseWithVectorResults(ifOp, ctx, loc, vecCondition);
  }

  if (hasElse || hasResults) return nullptr;

  Value vecCondition = ctx.valueMapping.lookupOrNull(condition);
  if (!vecCondition) {
    vecCondition = condition;
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

static LogicalResult handleConstantOp(arith::ConstantOp constOp, LoopVectorizationCtx &ctx) {
  cloneScalarOp(*constOp.getOperation(), ctx);
  return success();
}

static bool allOperandsNonNpuVector(Operation &op, LoopVectorizationCtx &ctx) {
  if (op.getNumOperands() == 0) return false;
  return llvm::all_of(op.getOperands(), [&](Value opnd) {
    Value v = ctx.valueMapping.lookupOrDefault(opnd);
    return !mlir::isa<npuvector::NPUVectorType>(v.getType());
  });
}

static LogicalResult handleArithOrMathOp(Operation &op, LoopVectorizationCtx &ctx) {
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

  if (hasIndexResult(op)) {
    cloneScalarOp(op, ctx);
    return success();
  }

  if (allOperandsNonNpuVector(op, ctx)) {
    cloneScalarOp(op, ctx);
    return success();
  }

  Value vecValue = vectorizeArithOp(&op, ctx);
  if (!vecValue) {
    return failure();
  }

  ctx.valueMapping.map(op.getResult(0), vecValue);

  return success();
}

static LogicalResult handleIfOp(scf::IfOp ifOp, LoopVectorizationCtx &ctx) {
  Value vecIfResult = vectorizeIf(ifOp, ctx);

  if (!vecIfResult && ifOp.getNumResults() > 0) {
    memref::StoreOp consumer = findUniqueStoreConsumer(ifOp);
    if (consumer && ctx.absorbedOps.contains(consumer.getOperation())) return success();
    return failure();
  }

  if (vecIfResult) {
    ctx.valueMapping.map(ifOp.getResult(0), vecIfResult);
  }

  return success();
}

static void updateNestedLoopOperands(scf::ForOp nestedForOp, LoopVectorizationCtx &ctx) {
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

static void mergeChildValueDimOrderIntoParent(const LoopVectorizationCtx &child, LoopVectorizationCtx &parentCtx) {
  for (const auto &kv : child.valueDimOrder) {
    if (parentCtx.valueDimOrder.find(kv.first) == parentCtx.valueDimOrder.end())
      parentCtx.valueDimOrder[kv.first] = kv.second;
  }
}

static void mergeAncestorValueDimOrderMissingKeys(LoopVectorizationCtx &ctx) {
  for (LoopVectorizationCtx *ancestor = ctx.parent; ancestor; ancestor = ancestor->parent) {
    for (const auto &kv : ancestor->valueDimOrder) {
      if (ctx.valueDimOrder.find(kv.first) == ctx.valueDimOrder.end()) ctx.valueDimOrder[kv.first] = kv.second;
    }
  }
}

static void registerChildResults(LoopVectorizationCtx &child, LoopVectorizationCtx &parentCtx) {
  if (child.mode == VectorizationMode::Elementwise) {
    for (const auto &kv : child.valueMapping.getValueMap()) {
      if (!parentCtx.valueMapping.lookupOrNull(kv.first)) parentCtx.valueMapping.map(kv.first, kv.second);
    }
    mergeChildValueDimOrderIntoParent(child, parentCtx);
    for (const auto &kv : child.allocBypass) parentCtx.allocBypass[kv.first] = kv.second;
    return;
  }

  if (child.mode == VectorizationMode::ReductionX) {
    mergeChildValueDimOrderIntoParent(child, parentCtx);
    return;
  }

  if (child.vecLoop) {
    for (auto [scalarResult, vecResult] : llvm::zip(child.scalarLoop.getResults(), child.vecLoop.getResults())) {
      parentCtx.valueMapping.map(scalarResult, vecResult);
      // finalizeReductionY / finalize may `replaceAllUsesWith(scalarResult, vecResult)` before
      // parent walks remaining ops; users then hold vecResult directly. Parent lookup must resolve
      // vecResult as well (IRMapping only had scalarResult -> vecResult as key).
      if (!parentCtx.valueMapping.lookupOrNull(vecResult)) parentCtx.valueMapping.map(vecResult, vecResult);
    }
    mergeChildValueDimOrderIntoParent(child, parentCtx);
  }
}

static LogicalResult handleNestedForOp(scf::ForOp nestedForOp, LoopVectorizationCtx &ctx) {
  int64_t nestedMaxStep = -1;
  VectorizationMode nestedMode = getVectorizationMode(nestedForOp, nestedMaxStep);

  if (nestedMode == VectorizationMode::None || nestedMaxStep <= 0) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  auto [skip, isDynamic] = checkLoopEligibility(nestedForOp);
  if (skip) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  int64_t childVecSize = computeStaticVectorSize(nestedForOp, nestedMaxStep);
  Value childMaxStep = ctx.builder.create<arith::ConstantIndexOp>(nestedForOp.getLoc(), nestedMaxStep);
  Value childVecSizeVal = isDynamic ? computeDynamicVectorSize(nestedForOp, childMaxStep, ctx.builder,
                                                               nestedForOp.getLoc(), &ctx.valueMapping)
                                    : Value();

  updateNestedLoopOperands(nestedForOp, ctx);

  LoopVectorizationCtx childCtx =
    LoopVectorizationCtx::createChild(ctx, nestedForOp, nestedMode, childVecSize, childVecSizeVal, childMaxStep);

  processLoop(childCtx);
  registerChildResults(childCtx, ctx);
  return success();
}

static LogicalResult vectorizeAllocTileBypass(memref::AllocOp allocOp, Operation &op, LoopVectorizationCtx &ctx) {
  auto memrefType = allocOp.getType();
  if (memrefType.hasStaticShape() && ctx.getRank() > 0) {
    auto shape = memrefType.getShape();
    bool matchesTile = (static_cast<int64_t>(shape.size()) == ctx.getRank());
    if (matchesTile) {
      for (unsigned i = 0; i < shape.size(); ++i) {
        if (shape[i] != ctx.allVectorSizes[i]) {
          matchesTile = false;
          break;
        }
      }
    }
    if (matchesTile) {
      ctx.allocBypass[allocOp.getResult()] = Value();
      cloneScalarOp(op, ctx);
      return success();
    }
  }
  cloneScalarOp(op, ctx);
  return success();
}

static LogicalResult vectorizeMemrefLoadLike(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
  Value memRef = loadOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);
  auto bypassIt = ctx.allocBypass.find(mappedMemRef);
  if (bypassIt == ctx.allocBypass.end()) bypassIt = ctx.allocBypass.find(memRef);
  if (bypassIt != ctx.allocBypass.end() && bypassIt->second) {
    ctx.valueMapping.map(loadOp.getResult(), bypassIt->second);
    return success();
  }

  Value vecValue = vectorizeLoad(loadOp, ctx);
  if (!vecValue) return failure();
  ctx.valueMapping.map(loadOp.getResult(), vecValue);
  return success();
}

static LogicalResult vectorizeMemrefStoreLike(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  Value memRef = storeOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);
  auto bypassIt = ctx.allocBypass.find(mappedMemRef);
  if (bypassIt == ctx.allocBypass.end()) bypassIt = ctx.allocBypass.find(memRef);
  if (bypassIt != ctx.allocBypass.end()) {
    Value storeValue = storeOp.getValue();
    Value vecVal = ctx.valueMapping.lookupOrNull(storeValue);
    if (vecVal) bypassIt->second = vecVal;
    return success();
  }

  if (!ctx.vf1FuncLevelNoAnchor) {
    bool hasVectorDim =
      llvm::any_of(storeOp.getIndices(), [&](Value idx) { return getVectorDimForIndex(idx, ctx) >= 0; });
    if (!hasVectorDim) {
      cloneScalarOp(*storeOp.getOperation(), ctx);
      return success();
    }
  }

  vectorizeStore(storeOp, ctx);
  return success();
}

static LogicalResult vectorizeOneOp(Operation &op, LoopVectorizationCtx &ctx) {
  if (ctx.absorbedOps.contains(&op)) return success();

  if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) return vectorizeAllocTileBypass(allocOp, op, ctx);

  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) return vectorizeMemrefLoadLike(loadOp, ctx);

  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) return vectorizeMemrefStoreLike(storeOp, ctx);

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

static LogicalResult vectorizeRegion(Region &region, LoopVectorizationCtx &ctx) {
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

static LogicalResult vectorizeLoopBody(LoopVectorizationCtx &ctx) {
  if (ctx.vecLoop) {
    ctx.builder.setInsertionPointToStart(ctx.vecLoop.getBody());
    ctx.valueMapping.map(ctx.scalarLoop.getInductionVar(), ctx.vecLoop.getInductionVar());
  }

  if (!ctx.allLoopToVectorDim.count(ctx.scalarLoop)) {
    const bool innerIvIsScalarForNestedVecAxis =
      (ctx.mode == VectorizationMode::ReductionY || ctx.mode == VectorizationMode::Broadcast) &&
      ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
    if (!innerIvIsScalarForNestedVecAxis) ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;
  }

  if (ctx.mode == VectorizationMode::ReductionX || ctx.mode == VectorizationMode::ReductionY) {
    for (auto [scalarArg, vecArg] : llvm::zip(ctx.scalarLoop.getRegionIterArgs(), ctx.vecLoop.getRegionIterArgs())) {
      ctx.valueMapping.map(scalarArg, vecArg);
    }
  }

  auto bodyOps = ctx.scalarLoop.getBody()->without_terminator();
  SmallVector<Operation *> opsToVectorize = llvm::map_to_vector(bodyOps, [](Operation &op) { return &op; });

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

  // Probe: nestedLoop.erase disabled to bisect pipeline malloc corruption; nested scf.for may
  // remain dead in IR — re-enable after root-cause fix.
  (void)nestedLoopsToErase;

  ensureReductionYield(ctx);
  return success();
}

static void ensureReductionYield(LoopVectorizationCtx &ctx) {
  if (!ctx.vecLoop || !ctx.scalarLoop.getInitArgs().empty()) return;
  if (ctx.mode != VectorizationMode::ReductionX && ctx.mode != VectorizationMode::ReductionY) return;
  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) return;
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(ctx.vecLoop.getLoc());
}

static void finalizeElementwise(LoopVectorizationCtx &ctx) {
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

static Value combineReductionResults(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                                     arith::AtomicRMWKind kind) {
  switch (kind) {
    case arith::AtomicRMWKind::addf:
      return builder.create<arith::AddFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::mulf:
      return builder.create<arith::MulFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::maximumf:
      return builder.create<arith::MaximumFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::minimumf:
      return builder.create<arith::MinimumFOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::addi:
      return builder.create<arith::AddIOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::muli:
      return builder.create<arith::MulIOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::maxs:
      return builder.create<arith::MaxSIOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::mins:
      return builder.create<arith::MinSIOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::maxu:
      return builder.create<arith::MaxUIOp>(loc, lhs, rhs);
    case arith::AtomicRMWKind::minu:
      return builder.create<arith::MinUIOp>(loc, lhs, rhs);
    default:
      return lhs;
  }
}

static SmallVector<Value> buildTailIndices(memref::LoadOp loadOp, Value inductionVar, Value tailStart,
                                           OpBuilder &builder) {
  SmallVector<Value> indices;
  for (Value idx : loadOp.getIndices()) {
    if (idx == inductionVar) {
      indices.push_back(tailStart);
    } else if (auto constOp = idx.getDefiningOp<arith::ConstantOp>()) {
      indices.push_back(builder.create<arith::ConstantOp>(constOp.getLoc(), constOp.getValue()));
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

static void vectorizeTailOps(LoopVectorizationCtx &tailCtx, LoopVectorizationCtx &ctx, Value vecLoopUb,
                             Value tailSize) {
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
          loadOp.getIndices(), [&](Value idx) { return tailCtx.valueMapping.lookupOrDefault(idx); });

        Value scalarLoad = tailCtx.builder.create<memref::LoadOp>(loc, mappedMemRef, indices);
        tailCtx.valueMapping.map(loadOp.getResult(), scalarLoad);
        continue;
      }

      SmallVector<Value> tailIndices =
        buildTailIndices(loadOp, ctx.scalarLoop.getInductionVar(), vecLoopUb, tailCtx.builder);

      Type memrefElemType = loadOp.getMemRefType().getElementType();
      auto typeInfo = createTailVectorType(tailSize, memrefElemType);

      Value padding = tailCtx.builder.create<arith::ConstantOp>(loc, tailCtx.builder.getZeroAttr(memrefElemType));

      SmallVector<Value> tailMaxSizes;
      tailMaxSizes.resize(typeInfo.dynamicSizes.size(), tailSize);
      auto tailRead = tailCtx.builder.create<npuvector::TransferReadOp>(
        loc, typeInfo.vecType, mappedMemRef, ValueRange(tailIndices), padding, Value(),
        ValueRange(typeInfo.dynamicSizes), ValueRange(tailMaxSizes));

      Value tailVecRes = tailRead.getResult();
      auto tailReadNvt = mlir::cast<npuvector::NPUVectorType>(tailVecRes.getType());
      const unsigned tailVecRank = static_cast<unsigned>(tailReadNvt.getRank());
      SmallVector<int> tailAxisOrder(tailVecRank);
      for (unsigned axisNum = 0; axisNum < tailVecRank; ++axisNum) tailAxisOrder[axisNum] = static_cast<int>(axisNum);
      reconcileValueDimOrderWithTileExtents(tailVecRes, tailAxisOrder, tailCtx);
      tailCtx.valueDimOrder[tailVecRes] = std::move(tailAxisOrder);
      tailCtx.valueMapping.map(loadOp.getResult(), tailVecRes);
      continue;
    }

    bool isIndexConst = tailLoadOp && llvm::any_of(tailLoadOp.getIndices(), [&](Value idx) {
                          return op.getNumResults() > 0 && op.getResult(0) == idx;
                        });
    if (isIndexConst) continue;

    if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
      Value scalarConst = tailCtx.builder.create<arith::ConstantOp>(constOp.getLoc(), constOp.getValue());
      tailCtx.valueMapping.map(constOp.getResult(), scalarConst);
    } else if (shouldCloneScalarOp(op, tailCtx)) {
      cloneScalarOp(op, tailCtx);
    } else if (op.getDialect()->getNamespace() == "arith" || op.getDialect()->getNamespace() == "math") {
      Value vecValue = vectorizeArithOp(&op, tailCtx);
      if (vecValue && op.getNumResults() > 0) {
        tailCtx.valueMapping.map(op.getResult(0), vecValue);
      }
    }
  }
}

static Value findValueToReduce(LoopVectorizationCtx &tailCtx, LoopVectorizationCtx &ctx, unsigned resultIdx) {
  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  if (resultIdx >= scalarYield.getNumOperands()) return Value();
  Value scalarYieldValue = scalarYield.getOperand(resultIdx);

  if (auto defOp = scalarYieldValue.getDefiningOp()) {
    if (isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp, arith::AddIOp, arith::MulIOp,
           arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp>(defOp)) {
      Value iterArg = ctx.scalarLoop.getRegionIterArgs()[resultIdx];
      for (Value operand : defOp->getOperands()) {
        if (operand != iterArg) {
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

static Value processTailBlock(LoopVectorizationCtx &ctx, Value reduced, Value vecLoopUb, Value tailSize, Type elemType,
                              vector::CombiningKind combiningKind, arith::AtomicRMWKind laneKind, unsigned resultIdx) {
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

  LoopVectorizationCtx tailCtx(ctx.builder, tailActualStep, tailVectorSizeValue, ctx.getMaxStepValue(), ctx.mode,
                               ctx.scalarLoop);
  tailCtx.parent = &ctx;
  for (const auto &kv : ctx.valueDimOrder) tailCtx.valueDimOrder[kv.first] = kv.second;
  vectorizeTailOps(tailCtx, ctx, vecLoopUb, tailSize);

  Value valueToReduce = findValueToReduce(tailCtx, ctx, resultIdx);
  if (!valueToReduce) {
    llvm_unreachable("Failed to find value to reduce in tail block");
  }

  auto tailReductionOp = tailCtx.builder.create<npuvector::ReductionOp>(loc, combiningKind, valueToReduce, Value(),
                                                                        arith::FastMathFlags::none);
  Value tailReduced = tailReductionOp.getDest();

  return combineReductionResults(tailCtx.builder, loc, reduced, tailReduced, laneKind);
}

/// True if any ancestor `scf.for` (walking from `loop`'s parent) carries a vectorization attr.
static bool parentHasTaggedVectorLoop(scf::ForOp loop) {
  for (Operation *parent = loop->getParentOp(); parent; parent = parent->getParentOp()) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) return true;
    }
  }
  return false;
}

static void finalizeReductionY(LoopVectorizationCtx &ctx) {
  Location loc = ctx.vecLoop.getLoc();

  if (ctx.scalarLoop.getInitArgs().empty()) {
    Block *body = ctx.vecLoop.getBody();
    if (body->empty() || !isa<scf::YieldOp>(body->back())) {
      ctx.builder.setInsertionPointToEnd(body);
      ctx.builder.create<scf::YieldOp>(loc);
    }

    ctx.builder.setInsertionPointAfter(ctx.vecLoop);

    if (!parentHasTaggedVectorLoop(ctx.scalarLoop)) {
      ctx.scalarLoop.erase();
    }
    return;
  }

  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  SmallVector<Value> vecYieldVals;
  vecYieldVals.reserve(scalarYield.getNumOperands());
  for (Value operand : scalarYield.getOperands()) {
    Value mapped = ctx.valueMapping.lookupOrNull(operand);
    if (!mapped) {
      return;
    }
    vecYieldVals.push_back(mapped);
  }

  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    body->back().erase();
  }
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(loc, vecYieldVals);

  for (auto [vecRes, yieldVal] : llvm::zip(ctx.vecLoop.getResults(), vecYieldVals)) {
    auto outNvt = mlir::dyn_cast<npuvector::NPUVectorType>(vecRes.getType());
    if (!outNvt) continue;
    auto ordIter = ctx.valueDimOrder.find(yieldVal);
    if (ordIter == ctx.valueDimOrder.end() || ordIter->second.empty()) continue;
    const unsigned wantRank = static_cast<unsigned>(outNvt.getRank());
    if (ordIter->second.size() != wantRank) continue;
    ctx.valueDimOrder[vecRes] = ordIter->second;
  }

  ctx.builder.setInsertionPointAfter(ctx.vecLoop);
  for (auto [scalarResult, vecResult] : llvm::zip(ctx.scalarLoop.getResults(), ctx.vecLoop.getResults())) {
    if (!scalarResult.use_empty()) {
      scalarResult.replaceAllUsesWith(vecResult);
    }
  }

  if (!parentHasTaggedVectorLoop(ctx.scalarLoop)) {
    ctx.scalarLoop.erase();
  }
}

static npuvector::NPUVectorType buildPartialReductionType(npuvector::NPUVectorType srcType, unsigned reductionDim) {
  auto srcShape = srcType.getShape();
  SmallVector<int64_t> resultShape;
  for (unsigned i = 0; i < srcShape.size(); ++i) {
    if (i != reductionDim) resultShape.push_back(srcShape[i]);
  }
  return npuvector::NPUVectorType::get(resultShape, srcType.getElementType());
}

static void vectorizeOpsForMultiDimTailCtx(scf::ForOp scalarLoop, LoopVectorizationCtx &tailCtx,
                                           memref::LoadOp &tailLoadOpOut) {
  tailLoadOpOut = nullptr;
  for (Operation &op : scalarLoop.getBody()->without_terminator()) {
    if (isReductionOp(op)) continue;
    if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
      vectorizeStore(storeOp, tailCtx);
      continue;
    }
    if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
      tailLoadOpOut = loadOp;
      Value vecValue = vectorizeLoad(loadOp, tailCtx);
      if (vecValue) tailCtx.valueMapping.map(loadOp.getResult(), vecValue);
      continue;
    }
    bool isIndexConst = tailLoadOpOut && llvm::any_of(tailLoadOpOut.getIndices(), [&](Value idx) {
                          return op.getNumResults() > 0 && op.getResult(0) == idx;
                        });
    if (isIndexConst) continue;
    if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
      Value val = tailCtx.builder.create<arith::ConstantOp>(constOp.getLoc(), constOp.getValue());
      tailCtx.valueMapping.map(constOp.getResult(), val);
    } else if (shouldCloneScalarOp(op, tailCtx)) {
      cloneScalarOp(op, tailCtx);
    } else if (op.getDialect()->getNamespace() == "arith" || op.getDialect()->getNamespace() == "math") {
      Value vecValue = vectorizeArithOp(&op, tailCtx);
      if (vecValue && op.getNumResults() > 0) tailCtx.valueMapping.map(op.getResult(0), vecValue);
    }
  }
}

static Value mergeMainReducedWithTail(Location loc, LoopVectorizationCtx &ctx, LoopVectorizationCtx &tailCtx,
                                      Value mainReduced, vector::CombiningKind combKind, unsigned resultIdx,
                                      arith::AtomicRMWKind laneKind) {
  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  Value scalarYieldVal = scalarYield.getOperand(resultIdx);
  Operation *defOp = scalarYieldVal.getDefiningOp();
  if (!defOp || !isa<arith::AddFOp, arith::MulFOp, arith::MaximumFOp, arith::MinimumFOp, arith::AddIOp, arith::MulIOp,
                     arith::MaxSIOp, arith::MinSIOp, arith::MaxUIOp, arith::MinUIOp>(defOp))
    return mainReduced;

  Value iterArgN = ctx.scalarLoop.getRegionIterArgs()[resultIdx];
  for (Value operand : defOp->getOperands()) {
    if (operand == iterArgN) continue;
    Value mapped = tailCtx.valueMapping.lookupOrNull(operand);
    if (!mapped || !mlir::isa<npuvector::NPUVectorType>(mapped.getType())) continue;

    auto tailVecType = mlir::cast<npuvector::NPUVectorType>(mapped.getType());
    auto tailReducedType = buildPartialReductionType(tailVecType, tailVecType.getRank() - 1);
    auto tailRedOp = tailCtx.builder.create<npuvector::ReductionOp>(
      loc, tailReducedType, combKind, mapped, Value(),
      tailCtx.builder.getDenseI64ArrayAttr({static_cast<int64_t>(tailVecType.getRank() - 1)}),
      arith::FastMathFlags::none);
    Value tailReduced = tailRedOp.getDest();

    switch (laneKind) {
    case arith::AtomicRMWKind::addf:
      return ctx.builder.create<arith::AddFOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::mulf:
      return ctx.builder.create<arith::MulFOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::maximumf:
      return ctx.builder.create<arith::MaximumFOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::minimumf:
      return ctx.builder.create<arith::MinimumFOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::addi:
      return ctx.builder.create<arith::AddIOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::muli:
      return ctx.builder.create<arith::MulIOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::maxs:
      return ctx.builder.create<arith::MaxSIOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::mins:
      return ctx.builder.create<arith::MinSIOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::maxu:
      return ctx.builder.create<arith::MaxUIOp>(loc, mainReduced, tailReduced);
    case arith::AtomicRMWKind::minu:
      return ctx.builder.create<arith::MinUIOp>(loc, mainReduced, tailReduced);
    default:
      return mainReduced;
    }
  }
  return mainReduced;
}

static Value processMultiDimTailBlock(LoopVectorizationCtx &ctx, LoopVectorizationCtx &parentCtx, Value mainReduced,
                                      Value vecLoopUb, Value tailSize, unsigned reductionDim,
                                      vector::CombiningKind combKind, unsigned resultIdx,
                                      arith::AtomicRMWKind laneKind) {
  Location loc = ctx.scalarLoop.getLoc();
  unsigned ancestorRank = ctx.allVectorSizes.size() - 1;

  SmallVector<int64_t> tailVecSizes(ctx.allVectorSizes.begin(), ctx.allVectorSizes.begin() + ancestorRank);
  SmallVector<Value> tailVecSizeValues(ctx.allVectorSizeValues.begin(), ctx.allVectorSizeValues.begin() + ancestorRank);
  SmallVector<Value> tailMaxSteps(ctx.allMaxStepValues.begin(), ctx.allMaxStepValues.begin() + ancestorRank);

  int64_t tailActualStep;
  Value tailVfValue;
  if (auto tailConst = tailSize.getDefiningOp<arith::ConstantIndexOp>()) {
    tailActualStep = tailConst.value();
    tailVfValue = nullptr;
  } else {
    tailActualStep = ctx.allVectorSizes.back();
    tailVfValue = tailSize;
  }
  tailVecSizes.push_back(tailActualStep);
  tailVecSizeValues.push_back(tailVfValue);
  tailMaxSteps.push_back(ctx.allMaxStepValues.back());

  LoopVectorizationCtx tailCtx(ctx.builder, tailVecSizes, tailVecSizeValues, tailMaxSteps,
                               VectorizationMode::ReductionX, ctx.scalarLoop);
  for (auto &[op, dim] : ctx.allLoopToVectorDim) tailCtx.allLoopToVectorDim[op] = dim;

  for (const auto &kv : parentCtx.valueMapping.getValueMap()) tailCtx.valueMapping.map(kv.first, kv.second);
  for (const auto &kv : ctx.valueMapping.getValueMap()) tailCtx.valueMapping.map(kv.first, kv.second);
  tailCtx.valueMapping.map(ctx.scalarLoop.getInductionVar(), vecLoopUb);

  tailCtx.parent = &ctx;
  for (const auto &kv : parentCtx.valueDimOrder) tailCtx.valueDimOrder[kv.first] = kv.second;
  for (const auto &kv : ctx.valueDimOrder) tailCtx.valueDimOrder[kv.first] = kv.second;

  memref::LoadOp tailLoadOp = nullptr;
  vectorizeOpsForMultiDimTailCtx(ctx.scalarLoop, tailCtx, tailLoadOp);

  return mergeMainReducedWithTail(loc, ctx, tailCtx, mainReduced, combKind, resultIdx, laneKind);
}

static LogicalResult inlineVectorize(LoopVectorizationCtx &ctx) {
  if (!ctx.parent) ctx.builder.setInsertionPoint(ctx.scalarLoop);

  if (!ctx.allLoopToVectorDim.count(ctx.scalarLoop)) ctx.allLoopToVectorDim[ctx.scalarLoop] = ctx.localDim;

  Value iv = ctx.scalarLoop.getInductionVar();
  Value lb =
    ctx.parent ? ctx.valueMapping.lookupOrDefault(ctx.scalarLoop.getLowerBound()) : ctx.scalarLoop.getLowerBound();
  ctx.valueMapping.map(iv, lb);

  Block *body = ctx.scalarLoop.getBody();
  SmallVector<Operation *> opsToVec;
  auto bodyOps = body->without_terminator();
  opsToVec.reserve(static_cast<size_t>(std::distance(bodyOps.begin(), bodyOps.end())));
  std::transform(bodyOps.begin(), bodyOps.end(), std::back_inserter(opsToVec),
                 [](Operation &singleOp) { return &singleOp; });

  SmallVector<scf::ForOp> nestedLoopsToErase;
  for (Operation *op : opsToVec) {
    if (!op || op->getBlock() == nullptr) continue;
    if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
      int64_t nm = -1;
      if (getVectorizationMode(nestedFor, nm) != VectorizationMode::None) nestedLoopsToErase.push_back(nestedFor);
    }
    if (failed(vectorizeOneOp(*op, ctx))) return failure();
  }

  // Probe: nestedLoop.erase disabled (see vectorizeLoopBody); re-enable after bisect.
  (void)nestedLoopsToErase;

  if (!ctx.parent) {
    ctx.scalarLoop.erase();
  }
  return success();
}

static Value applyReductionXTripTail(Location loc, LoopVectorizationCtx &ctx, Value reduced, Value vecLoopUb,
                                     Type elemType, bool isMultiDim, unsigned reductionDim,
                                     vector::CombiningKind combKind, unsigned resultIdx,
                                     arith::AtomicRMWKind laneKind) {
  Value originalUb = ctx.scalarLoop.getUpperBound();
  Value finalResult = reduced;
  LoopVectorizationCtx &parentOrSelf = ctx.parent ? *ctx.parent : ctx;

  std::optional<int64_t> origUbOpt = tryConstantIndex(originalUb);
  std::optional<int64_t> lbTailOpt = tryConstantIndex(ctx.scalarLoop.getLowerBound());
  if (origUbOpt && lbTailOpt) {
    int64_t tripCount = *origUbOpt - *lbTailOpt;
    int64_t reductionStep = ctx.allVectorSizes.back();
    int64_t remainder = tripCount % reductionStep;
    if (remainder != 0) {
      Value tailSize = ctx.builder.create<arith::ConstantIndexOp>(loc, remainder);
      finalResult = isMultiDim ? processMultiDimTailBlock(ctx, parentOrSelf, reduced, vecLoopUb, tailSize,
                                                          reductionDim, combKind, resultIdx, laneKind)
                               : processTailBlock(ctx, reduced, vecLoopUb, tailSize, elemType, combKind, laneKind,
                                                  resultIdx);
    }
    return finalResult;
  }

  Value lb = ctx.scalarLoop.getLowerBound();
  Value tripCount = ctx.builder.create<arith::SubIOp>(loc, originalUb, lb);
  Value vfVal = ctx.allVectorSizeValues.back()
                  ? ctx.allVectorSizeValues.back()
                  : ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes.back());
  Value remainder = ctx.builder.create<arith::RemSIOp>(loc, tripCount, vfVal);
  Value c0 = ctx.builder.create<arith::ConstantIndexOp>(loc, 0);
  Value needTail = ctx.builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, remainder, c0);

  Type ifResultType = isMultiDim ? reduced.getType() : TypeRange{elemType}.front();
  auto ifOp = ctx.builder.create<scf::IfOp>(loc, TypeRange{ifResultType}, needTail, true);
  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(ifOp.thenBlock());
    Value tailResult =
      isMultiDim ? processMultiDimTailBlock(ctx, parentOrSelf, reduced, vecLoopUb, remainder, reductionDim, combKind,
                                            resultIdx, laneKind)
                 : processTailBlock(ctx, reduced, vecLoopUb, remainder, elemType, combKind, laneKind, resultIdx);
    ctx.builder.create<scf::YieldOp>(loc, tailResult);
  }
  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(ifOp.elseBlock());
    ctx.builder.create<scf::YieldOp>(loc, reduced);
  }
  return ifOp.getResult(0);
}

static void mergeReductionXOriginalInit(Location loc, LoopVectorizationCtx &ctx, bool isMultiDim, unsigned totalRank,
                                        Value &finalResult, unsigned laneIdx) {
  if (laneIdx >= ctx.reductionKinds.size() || laneIdx >= ctx.origInits.size()) return;
  arith::AtomicRMWKind laneKind = ctx.reductionKinds[laneIdx];
  Value laneOrig = ctx.origInits[laneIdx];
  if (isNeutralElement(laneKind, laneOrig, ctx.builder)) return;
  if (isMultiDim) {
    unsigned ancestorRank = totalRank - 1;
    if (ancestorRank > 0) {
      SmallVector<int64_t> ancVecSizes(ctx.allVectorSizes.begin(), ctx.allVectorSizes.begin() + ancestorRank);
      SmallVector<Value> ancVecSizeValues(ctx.allVectorSizeValues.begin(),
                                          ctx.allVectorSizeValues.begin() + ancestorRank);
      SmallVector<Value> ancMaxSteps(ctx.allMaxStepValues.begin(), ctx.allMaxStepValues.begin() + ancestorRank);
      LoopVectorizationCtx ancCtx(ctx.builder, ancVecSizes, ancVecSizeValues, ancMaxSteps,
                                  VectorizationMode::Elementwise, ctx.scalarLoop);
      ancCtx.parent = &ctx;
      for (const auto &kv : ctx.valueMapping.getValueMap()) ancCtx.valueMapping.map(kv.first, kv.second);
      for (const auto &kv : ctx.valueDimOrder) ancCtx.valueDimOrder[kv.first] = kv.second;
      Value initBroadcast = vectorizeBroadcastScalar(laneOrig, ancCtx);
      if (initBroadcast)
        finalResult = combineReductionResults(ctx.builder, loc, finalResult, initBroadcast, laneKind);
    } else {
      finalResult = combineReductionResults(ctx.builder, loc, finalResult, laneOrig, laneKind);
    }
  } else {
    finalResult = combineReductionResults(ctx.builder, loc, finalResult, laneOrig, laneKind);
  }
}

static void finalizeReductionXOutputs(LoopVectorizationCtx &ctx, ArrayRef<Value> finalResults) {
  if (ctx.parent) {
    for (auto [scalarResult, finalResult] : llvm::zip(ctx.scalarLoop.getResults(), finalResults)) {
      ctx.parent->valueMapping.map(scalarResult, finalResult);
      if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(finalResult.getType())) {
        if (nvt.getRank() > 1) {
          SmallVector<int> parentDimOrder;
          for (auto &[loop, localDim] : ctx.allLoopToVectorDim) {
            if (loop == ctx.scalarLoop.getOperation()) continue;
            auto pit = ctx.parent->allLoopToVectorDim.find(loop);
            if (pit != ctx.parent->allLoopToVectorDim.end()) parentDimOrder.push_back(static_cast<int>(pit->second));
          }
          llvm::sort(parentDimOrder);
          ctx.parent->valueDimOrder[finalResult] = parentDimOrder;
        }
      }
    }
  } else {
    for (auto [scalarResult, finalResult] : llvm::zip(ctx.scalarLoop.getResults(), finalResults)) {
      if (!scalarResult.use_empty()) scalarResult.replaceAllUsesWith(finalResult);
    }
    ctx.scalarLoop.erase();
  }
}

static LogicalResult reductionXVectorize(LoopVectorizationCtx &ctx) {
  if (!ctx.parent) ctx.builder.setInsertionPoint(ctx.scalarLoop);
  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) return failure();
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }

  Location loc = ctx.vecLoop.getLoc();
  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  const unsigned numResults = scalarYield.getNumOperands();
  if (numResults == 0 || numResults != ctx.vecLoop.getNumResults() ||
      numResults != ctx.reductionKinds.size())
    return failure();

  SmallVector<Value> vecYieldVals;
  vecYieldVals.reserve(numResults);
  for (unsigned idx = 0; idx < numResults; ++idx) {
    Value mapped = ctx.valueMapping.lookupOrNull(scalarYield.getOperand(idx));
    if (!mapped) return failure();
    vecYieldVals.push_back(mapped);
  }

  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) body->back().erase();
  ctx.builder.setInsertionPointToEnd(body);
  ctx.builder.create<scf::YieldOp>(loc, vecYieldVals);
  ctx.builder.setInsertionPointAfter(ctx.vecLoop);

  SmallVector<Value> finalResults;
  finalResults.reserve(numResults);

  for (unsigned idx = 0; idx < numResults; ++idx) {
    arith::AtomicRMWKind laneKind = ctx.reductionKinds[idx];
    vector::CombiningKind combKind = convertToCombiningKind(laneKind);
    Value vectorAcc = ctx.vecLoop.getResult(idx);
    auto fullVecType = mlir::cast<npuvector::NPUVectorType>(vectorAcc.getType());
    Type elemType = fullVecType.getElementType();
    const unsigned totalRank = fullVecType.getRank();

    const bool isMultiDim = totalRank > 1;
    const unsigned reductionDim = totalRank - 1;

    Value reduced;
    if (isMultiDim) {
      auto reducedVecType = buildPartialReductionType(fullVecType, reductionDim);
      auto reductionOp = ctx.builder.create<npuvector::ReductionOp>(
        loc, reducedVecType, combKind, vectorAcc, Value(),
        ctx.builder.getDenseI64ArrayAttr({static_cast<int64_t>(reductionDim)}), arith::FastMathFlags::none);
      reduced = reductionOp.getDest();
    } else {
      auto reductionOp =
        ctx.builder.create<npuvector::ReductionOp>(loc, combKind, vectorAcc, Value(), arith::FastMathFlags::none);
      reduced = reductionOp.getDest();
    }

    Value vecLoopUb = ctx.vecLoop.getUpperBound();
    Value finalResult =
      applyReductionXTripTail(loc, ctx, reduced, vecLoopUb, elemType, isMultiDim, reductionDim, combKind, idx,
                              laneKind);

    mergeReductionXOriginalInit(loc, ctx, isMultiDim, totalRank, finalResult, idx);
    finalResults.push_back(finalResult);
  }

  finalizeReductionXOutputs(ctx, finalResults);

  return success();
}

static LogicalResult reductionYVectorize(LoopVectorizationCtx &ctx) {
  const bool innerIvIsScalar = ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
  if (!innerIvIsScalar) ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;

  if (ctx.vectorizationAxis) {
    Value outerIVMapping = ctx.valueMapping.lookupOrDefault(ctx.vectorizationAxis);
    if (outerIVMapping && outerIVMapping != ctx.vectorizationAxis)
      ctx.valueMapping.map(ctx.vectorizationAxis, outerIVMapping);
  }

  if (!ctx.builder.getInsertionBlock()) ctx.builder.setInsertionPoint(ctx.scalarLoop);

  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) return failure();
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }
  finalizeReductionY(ctx);
  return success();
}

static LogicalResult broadcastVectorize(LoopVectorizationCtx &ctx) {
  const bool innerIvIsScalar = ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
  if (!innerIvIsScalar) ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;

  if (ctx.vectorizationAxis) {
    Value outerIVMapping = ctx.valueMapping.lookupOrDefault(ctx.vectorizationAxis);
    if (outerIVMapping && outerIVMapping != ctx.vectorizationAxis)
      ctx.valueMapping.map(ctx.vectorizationAxis, outerIVMapping);
  }

  if (!ctx.builder.getInsertionBlock()) ctx.builder.setInsertionPoint(ctx.scalarLoop);

  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) return failure();
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }
  finalizeElementwise(ctx);
  return success();
}

static void processLoop(LoopVectorizationCtx &ctx) {
  switch (ctx.mode) {
    case VectorizationMode::Elementwise:
      (void)inlineVectorize(ctx);
      break;
    case VectorizationMode::ReductionX:
      (void)reductionXVectorize(ctx);
      break;
    case VectorizationMode::ReductionY:
      (void)reductionYVectorize(ctx);
      break;
    case VectorizationMode::Broadcast:
      (void)broadcastVectorize(ctx);
      break;
    default:
      break;
  }
}

static LoopVectorizationCtx createVF1SweepCtx(OpBuilder &builder, Location loc) {
  Value maxStepConst = builder.create<arith::ConstantIndexOp>(loc, 1);
  LoopVectorizationCtx ctx(builder, /*actualStep*/ 1, /*vfVal*/ nullptr, maxStepConst, VectorizationMode::Elementwise,
                           scf::ForOp(), nullptr);
  ctx.localDim = 0;
  ctx.vf1FuncLevelNoAnchor = true;
  return ctx;
}

static void expandVF1ForwardClosure(memref::LoadOp root, DenseSet<Operation *> &closure) {
  closure.clear();
  SmallVector<Operation *> stack;
  stack.push_back(root);
  while (!stack.empty()) {
    Operation *op = stack.pop_back_val();
    if (!closure.insert(op).second) continue;

    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (llvm::any_of(user->getResults(), [](Value v) {
              return mlir::isa<npuvector::NPUVectorType>(v.getType());
            }))
          continue;
        if (mlir::isa<scf::YieldOp>(user))
          continue;
        stack.push_back(user);
      }
    }
  }
}

static LogicalResult topoSortVF1Closure(const DenseSet<Operation *> &closureSet,
                                          SmallVectorImpl<Operation *> &topoOut) {
  DenseMap<Operation *, unsigned> indegree;
  for (Operation *op : closureSet) indegree[op] = 0;
  for (Operation *op : closureSet) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && closureSet.contains(def)) indegree[op]++;
    }
  }

  std::deque<Operation *> ready;
  std::copy_if(closureSet.begin(), closureSet.end(), std::back_inserter(ready),
               [&indegree](Operation *op) { return indegree[op] == 0; });

  topoOut.clear();
  while (!ready.empty()) {
    Operation *op = ready.front();
    ready.pop_front();
    topoOut.push_back(op);
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (!closureSet.contains(user)) continue;
        if (--indegree[user] == 0) ready.push_back(user);
      }
    }
  }

  if (topoOut.size() != closureSet.size()) return failure();
  return success();
}

enum class Vf1ChainPromotionResult { Promoted, Skipped, FatalError };

static Vf1ChainPromotionResult tryPromoteVF1Chain(memref::LoadOp rootLoad) {
  DenseSet<Operation *> closure;
  expandVF1ForwardClosure(rootLoad, closure);

  SmallVector<Operation *> topo;
  if (failed(topoSortVF1Closure(closure, topo))) return Vf1ChainPromotionResult::Skipped;


  OpBuilder builder(rootLoad);
  LoopVectorizationCtx ctx = createVF1SweepCtx(builder, rootLoad.getLoc());

  for (Operation *op : topo) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);

    if (auto ld = dyn_cast<memref::LoadOp>(op)) {
      Value vecLoaded = vectorizeLoad(ld, ctx);
      if (!vecLoaded) {
        emitError(ld.getLoc()) << "npuvector-vectorize: VF1 sweep vectorizeLoad did not produce a value";
        return Vf1ChainPromotionResult::FatalError;
      }
      ld.getResult().replaceAllUsesWith(vecLoaded);
      ld.erase();
      continue;
    }
    if (auto st = dyn_cast<memref::StoreOp>(op)) {
      vectorizeStore(st, ctx);
      st.erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      cloneScalarOp(*op, ctx);
      Value replacement = ctx.valueMapping.lookup(op->getResult(0));
      op->getResult(0).replaceAllUsesWith(replacement);
      op->erase();
      continue;
    }

    if (shouldCloneScalarOp(*op, ctx)) {
      cloneScalarOp(*op, ctx);
      Value replacement = ctx.valueMapping.lookup(op->getResult(0));
      op->getResult(0).replaceAllUsesWith(replacement);
      op->erase();
      continue;
    }

    if (op->getNumResults() == 0) {
      emitError(op->getLoc())
        << "npuvector-vectorize: VF1 sweep cannot vectorize an op with no results (use a clone path)";
      return Vf1ChainPromotionResult::FatalError;
    }

    Value vecVal = vectorizeArithOp(op, ctx);
    if (!vecVal) return Vf1ChainPromotionResult::FatalError;
    op->getResult(0).replaceAllUsesWith(vecVal);
    op->erase();
  }
  return Vf1ChainPromotionResult::Promoted;
}

static Vf1ChainPromotionResult tryPromoteVF1Store(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  if (storeOp.getIndices().empty()) return Vf1ChainPromotionResult::Skipped;

  OpBuilder::InsertionGuard guard(ctx.builder);
  ctx.builder.setInsertionPoint(storeOp);
  Value storeValue = storeOp.getValue();
  vectorizeStore(storeOp, ctx);
  ctx.valueMapping.erase(storeValue);
  storeOp.erase();
  return Vf1ChainPromotionResult::Promoted;
}

static LogicalResult runMemRefLoadVF1Sweep(func::FuncOp funcOp) {
  constexpr unsigned kMaxRounds = 64;
  for (unsigned round = 0; round < kMaxRounds; ++round) {
    bool changed = false;
    SmallVector<memref::LoadOp> loads;
    funcOp.walk([&](memref::LoadOp ld) { loads.push_back(ld); });

    for (memref::LoadOp ld : loads) {
      Vf1ChainPromotionResult outcome = tryPromoteVF1Chain(ld);
      if (outcome == Vf1ChainPromotionResult::FatalError) return failure();
      if (outcome == Vf1ChainPromotionResult::Promoted) changed = true;
    }
    if (!changed) break;
  }
  return success();
}

static LogicalResult runMemRefStoreVF1Sweep(func::FuncOp funcOp) {
  SmallVector<memref::StoreOp> stores;
  funcOp.walk([&](memref::StoreOp storeOp) {
    if (!storeOp.getIndices().empty()) stores.push_back(storeOp);
  });
  if (stores.empty()) return success();

  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  LoopVectorizationCtx ctx = createVF1SweepCtx(builder, funcOp.getLoc());
  for (memref::StoreOp storeOp : stores) {
    Vf1ChainPromotionResult outcome = tryPromoteVF1Store(storeOp, ctx);
    if (outcome == Vf1ChainPromotionResult::FatalError) return failure();
  }
  return success();
}

static void runVectorization(scf::ForOp loop, VectorizationMode mode, OpBuilder &builder, int64_t maxStepFromAttr,
                             bool isDynamic) {
  OpBuilder attrBuilder(builder.getContext());
  attrBuilder.setInsertionPoint(loop);

  int64_t actualStep = computeStaticVectorSize(loop, maxStepFromAttr);
  Value maxStepValue = attrBuilder.create<arith::ConstantIndexOp>(loop.getLoc(), maxStepFromAttr);
  Value vectorSizeValue =
    isDynamic ? computeDynamicVectorSize(loop, maxStepValue, attrBuilder, loop.getLoc(), nullptr) : Value();

  SmallVector<int64_t> vecSizes = {actualStep};
  SmallVector<Value> vecSizeValues = {vectorSizeValue};
  SmallVector<Value> maxSteps = {maxStepValue};
  LoopVectorizationCtx ctx(builder, vecSizes, vecSizeValues, maxSteps, mode, loop, loop.getInductionVar());

  const bool innerIvIsScalar = (mode == VectorizationMode::ReductionY || mode == VectorizationMode::Broadcast);
  if (!innerIvIsScalar) {
    ctx.allLoopToVectorDim[loop] = 0;
    ctx.localDim = 0;
  }
  processLoop(ctx);
}

class NPUVectorVectorizePass : public mlir::scf::impl::NPUVectorVectorizePassBase<NPUVectorVectorizePass> {
 public:
  NPUVectorVectorizePass() = default;
  NPUVectorVectorizePass(const NPUVectorVectorizePass &) = default;

  StringRef getArgument() const override { return "npuvector-vectorize"; }

  StringRef getDescription() const override {
    return "SCF loop vectorization using NPUVector with dynamic shape support";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, npuvector::NPUVectorDialect, memref::MemRefDialect, func::FuncDialect,
                    arith::ArithDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hacc::utils::isDevice(funcOp)) return;

    OpBuilder builder(&getContext());

    SmallVector<scf::ForOp> allCandidateLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (hasVectorizationAttr(forOp)) allCandidateLoops.push_back(forOp);
    });

    SmallVector<scf::ForOp> topLevelLoops;
    for (scf::ForOp loop : allCandidateLoops) {
      bool isNested = false;
      for (Operation *parent = loop->getParentOp(); parent && !isa<func::FuncOp>(parent);
           parent = parent->getParentOp()) {
        if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
          if (hasVectorizationAttr(parentFor)) {
            isNested = true;
            break;
          }
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

      runVectorization(loop, mode, builder, maxStepFromAttr, isDynamic);
    }

    if (failed(runMemRefLoadVF1Sweep(funcOp)) || failed(runMemRefStoreVF1Sweep(funcOp))) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::scf::createNPUVectorVectorizePass() {
  return std::make_unique<NPUVectorVectorizePass>();
}
