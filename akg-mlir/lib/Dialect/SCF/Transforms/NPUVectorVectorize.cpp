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
//      └── vectorizeBroadcastScalar → npuvector.broadcast (rank-lift `dimension` must resolve; no fallback)
//   4. Finalization
//      ├── Elementwise: inline or loop-based transformation
//      └── Reduction: vector reduction + tail processing + init value merging
//   5. Phase 2 (same pass tail): rank-0 sweep — fictional LoopVectorizationCtx instances (Elementwise, no scf.for
//      anchor, vf1FuncLevelNoAnchor) over remaining mixed arith/math ops, scalar memref.load chains, and
//      store-rooted reduction consumers.
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
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallBitVector.h"
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
#include "akg/Utils/Constants.h"

namespace mlir {
namespace scf {
#define GEN_PASS_DECL_NPUVECTORVECTORIZE
#define GEN_PASS_DEF_NPUVECTORVECTORIZE
#include "akg/Dialect/SCF/Passes.h.inc"

}  // namespace scf
}  // namespace mlir

namespace {

constexpr int kMaxMemrefAccessNormalizeGuards = 64;
constexpr size_t kMinMergedReductionLoopCount = 2;
using namespace mlir;  // NOLINT(build/namespaces)

enum class VectorizationMode {
  None,
  Elementwise,
  ReductionX,
  ReductionY,
  /// Inner loop: IV stays scalar; mem ops vectorize along parent axis (like nested
  /// reduction_y mapping) but no reduction iter_args — finalize like elementwise.
  Broadcast,
};

struct ScratchTileMeta {
  /// Store-time valueDimOrder (tile layout). Prefer tileDimToLoop for cross-nest rebind.
  SmallVector<int> tileAxisOrder;
  /// Parallel to tileAxisOrder: alloc memref dim for this tile dim, or -1 if SSA-only (shadow).
  SmallVector<int> allocDimOfTileDim;
  /// Parallel: scf.for Operation* that owned the axis at store time (for shadow rebind).
  SmallVector<Operation *> tileDimToLoop;
  /// Parallel: store IV lower bound for alloc-backed dims (0 for shadow).
  SmallVector<int64_t> storeLowerBoundOfTileDim;
  /// Each entry is (allocDim, tileGlobalAxis, loopLowerBound). Kept for empty-channel full-tile path.
  SmallVector<std::tuple<int, int, int64_t>> channelDims;
  SmallVector<OpFoldResult> storeRootIndices;
  int allocRank = 0;
};

struct CurrentTileSliceInfo {
  Value fullVector;
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
  SmallVector<int> dimToAxis;
};

struct LoopVectorizationInit {
  OpBuilder &builder;
  SmallVector<int64_t> vecSizes;
  SmallVector<Value> vecSizeVals;
  SmallVector<Value> maxVals;
  VectorizationMode mode = VectorizationMode::None;
  mutable scf::ForOp loop;
  Value vecAxis = nullptr;
};

struct LoopVectorizationScalarInit {
  OpBuilder &builder;
  int64_t actualStepVal = 0;
  Value vfVal;
  Value maxVal;
  VectorizationMode mode = VectorizationMode::None;
  mutable scf::ForOp loop;
  Value vecAxis = nullptr;
};

struct LoopVectorizationCtx;

struct CreateChildParams {
  LoopVectorizationCtx &parentCtx;
  mutable scf::ForOp childLoop;
  VectorizationMode childMode = VectorizationMode::None;
  int64_t childVecSize = 0;
  Value childVecSizeVal;
  Value childMaxStepVal;
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
  SmallVector<Value> currentVectorSizeValues;
  DenseMap<Operation *, unsigned> allLoopToVectorDim;
  SmallVector<Operation *> mergedReductionLoops;

  IRMapping valueMapping;
  DenseMap<Value, SmallVector<int>> valueDimOrder;

  /// Per `iter_arg` / loop result for ReductionX/Y (order matches `scalarLoop` init args).
  SmallVector<arith::AtomicRMWKind> reductionKinds;
  SmallVector<Value> origInits;
  DenseMap<unsigned, CurrentTileSliceInfo> iterArgTileSlices;

  Value vectorizationAxis;
  std::optional<unsigned> vectorAxisVectorDim;

  DenseSet<Operation *> absorbedOps;

  DenseMap<Value, Value> allocBypass;

  DenseMap<Value, ScratchTileMeta> scratchMeta;

  SmallVector<std::tuple<Value, SmallVector<int64_t>, Value>> scratchSliceCache;

  bool vf1FuncLevelNoAnchor = false;

  explicit LoopVectorizationCtx(LoopVectorizationInit init)
      : builder(init.builder),
        scalarLoop(init.loop),
        vecLoop(nullptr),
        mode(init.mode),
        allVectorSizes(std::move(init.vecSizes)),
        allVectorSizeValues(std::move(init.vecSizeVals)),
        allMaxStepValues(std::move(init.maxVals)),
        currentVectorSizeValues(allVectorSizes.size()),
        vectorizationAxis(init.vecAxis) {}

  explicit LoopVectorizationCtx(LoopVectorizationScalarInit init)
      : builder(init.builder),
        scalarLoop(init.loop),
        vecLoop(nullptr),
        mode(init.mode),
        allVectorSizes({init.actualStepVal}),
        allVectorSizeValues({init.vfVal}),
        allMaxStepValues({init.maxVal}),
        currentVectorSizeValues(allVectorSizes.size()),
        vectorizationAxis(init.vecAxis) {}

  [[nodiscard]] int64_t getRank() const { return allVectorSizes.size(); }

  [[nodiscard]] bool isDynamic() const {
    return llvm::any_of(allVectorSizeValues, [](Value v) { return v != nullptr; });
  }

  [[nodiscard]] int64_t getActualStep() const { return allVectorSizes.back(); }
  [[nodiscard]] Value getVectorSizeValue() const { return allVectorSizeValues.back(); }
  [[nodiscard]] Value getMaxStepValue() const { return allMaxStepValues.back(); }

  Value getVectorizationAxis() {
    if (vectorizationAxis && (mode == VectorizationMode::ReductionY || mode == VectorizationMode::Broadcast)) {
      return vectorizationAxis;
    }
    if (scalarLoop.getOperation() == nullptr) {
      return {};
    }
    return scalarLoop.getInductionVar();
  }

  [[nodiscard]] npuvector::NPUVectorType getVectorType(Type elemType) const {
    SmallVector<int64_t> shape;
    for (unsigned i = 0; i < allVectorSizes.size(); ++i) {
      if (allVectorSizeValues[i]) {
        shape.push_back(ShapedType::kDynamic);
      } else {
        shape.push_back(allVectorSizes[i]);
      }
    }
    return npuvector::NPUVectorType::get(shape, elemType);
  }

  [[nodiscard]] int getVectorDimForIV(Value iv) const {
    for (auto &[op, dim] : allLoopToVectorDim) {
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (forOp.getInductionVar() == iv) {
          return static_cast<int>(dim);
        }
      }
    }
    if (vectorizationAxis && iv == vectorizationAxis && vectorAxisVectorDim) {
      return static_cast<int>(*vectorAxisVectorDim);
    }
    return -1;
  }

  static LoopVectorizationCtx createChild(const CreateChildParams &params) {
    LoopVectorizationCtx &parentCtx = params.parentCtx;
    scf::ForOp childLoop = params.childLoop;
    SmallVector<int64_t> emptyS;
    SmallVector<Value> emptyV;
    LoopVectorizationCtx child(LoopVectorizationInit{parentCtx.builder, std::move(emptyS), std::move(emptyV),
                                                     std::move(emptyV), params.childMode, childLoop});
    child.parent = &parentCtx;

    SmallVector<std::pair<Operation *, unsigned>> ancestorLoops;
    for (Operation *p = childLoop->getParentOp(); p != nullptr; p = p->getParentOp()) {
      auto it = parentCtx.allLoopToVectorDim.find(p);
      if (it != parentCtx.allLoopToVectorDim.end()) {
        ancestorLoops.push_back({it->first, it->second});
      }
    }
    llvm::sort(ancestorLoops,
               [](const auto &pairLeft, const auto &pairRight) { return pairLeft.second < pairRight.second; });

    DenseMap<unsigned, unsigned> parentDimToLocalDim;
    for (auto &[loop, parentDim] : ancestorLoops) {
      unsigned lDim = child.allVectorSizes.size();
      parentDimToLocalDim[parentDim] = lDim;
      child.allVectorSizes.push_back(parentCtx.allVectorSizes[parentDim]);
      int64_t maxStepInt = parentCtx.allVectorSizes[parentDim];
      if (auto cst = parentCtx.allMaxStepValues[parentDim].getDefiningOp<arith::ConstantIndexOp>()) {
        maxStepInt = cst.value();
      }
      child.allMaxStepValues.push_back(
        parentCtx.builder.create<arith::ConstantIndexOp>(childLoop.getLoc(), maxStepInt));
      Value vfVal = nullptr;
      if (parentCtx.allVectorSizeValues[parentDim]) {
        vfVal = parentCtx.valueMapping.lookupOrDefault(parentCtx.allVectorSizeValues[parentDim]);
      }
      child.allVectorSizeValues.push_back(vfVal);
      Value currentVfVal = nullptr;
      if (parentDim < parentCtx.currentVectorSizeValues.size() && parentCtx.currentVectorSizeValues[parentDim]) {
        currentVfVal = parentCtx.valueMapping.lookupOrDefault(parentCtx.currentVectorSizeValues[parentDim]);
      }
      child.currentVectorSizeValues.push_back(currentVfVal);
      child.allLoopToVectorDim[loop] = lDim;
    }

    bool addsVecDim =
      (params.childMode == VectorizationMode::Elementwise || params.childMode == VectorizationMode::ReductionX);
    if (addsVecDim) {
      child.localDim = child.allVectorSizes.size();
      child.allVectorSizes.push_back(params.childVecSize);
      child.allVectorSizeValues.push_back(params.childVecSizeVal);
      child.allMaxStepValues.push_back(params.childMaxStepVal);
      child.currentVectorSizeValues.push_back(nullptr);
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

    for (const auto &kv : parentCtx.valueMapping.getValueMap()) {
      child.valueMapping.map(kv.first, kv.second);
    }

    for (const auto &kv : parentCtx.valueDimOrder) {
      child.valueDimOrder[kv.first] = kv.second;
    }

    for (const auto &kv : parentCtx.allocBypass) {
      child.allocBypass[kv.first] = kv.second;
    }
    for (const auto &kv : parentCtx.scratchMeta) {
      child.scratchMeta[kv.first] = kv.second;
    }

    child.mergedReductionLoops = parentCtx.mergedReductionLoops;
    return child;
  }

  static LoopVectorizationCtx createMergedReductionChild(LoopVectorizationCtx &parentCtx, scf::ForOp childLoop) {
    LoopVectorizationCtx child(LoopVectorizationInit{parentCtx.builder, parentCtx.allVectorSizes,
                                                     parentCtx.allVectorSizeValues, parentCtx.allMaxStepValues,
                                                     VectorizationMode::ReductionX, childLoop});
    child.parent = &parentCtx;
    child.localDim = parentCtx.allLoopToVectorDim.lookup(childLoop);
    child.allLoopToVectorDim = parentCtx.allLoopToVectorDim;
    child.currentVectorSizeValues = parentCtx.currentVectorSizeValues;
    child.mergedReductionLoops = parentCtx.mergedReductionLoops;

    for (const auto &kv : parentCtx.valueMapping.getValueMap()) {
      child.valueMapping.map(kv.first, kv.second);
    }
    for (const auto &kv : parentCtx.valueDimOrder) {
      child.valueDimOrder[kv.first] = kv.second;
    }
    for (const auto &kv : parentCtx.allocBypass) {
      child.allocBypass[kv.first] = kv.second;
    }
    for (const auto &kv : parentCtx.scratchMeta) {
      child.scratchMeta[kv.first] = kv.second;
    }
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
static Value getRuntimeVectorSizeValue(LoopVectorizationCtx &ctx, unsigned axisIdx, Location loc);

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
  if (valueMapping != nullptr) {
    upperBound = valueMapping->lookupOrDefault(upperBound);
    lowerBound = valueMapping->lookupOrDefault(lowerBound);
  }

  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeThree> operands{upperBound, lowerBound, maxStepValue};
  SmallVector<AffineExpr, kSmallVectorSizeTwo> minExprs{getAffineDimExpr(0, context) - getAffineDimExpr(1, context),
                                                        getAffineDimExpr(2, context)};
  AffineMap minMap = AffineMap::get(kSmallVectorSizeThree, kSmallVectorSizeZero, minExprs, context);
  return builder.create<affine::AffineMinOp>(loc, minMap, operands).getResult();
}

static VectorizationMode getVectorizationMode(scf::ForOp loop, int64_t &maxStepFromAttr) {
  if (loop->hasAttr(kReductionXLoopAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kReductionXLoopAttr);
    if (!attr) {
      return VectorizationMode::None;
    }
    maxStepFromAttr = attr.getInt();
    return VectorizationMode::ReductionX;
  }

  if (loop->hasAttr(kReductionYLoopAttr)) {
    if (auto attr = loop->getAttrOfType<IntegerAttr>(kReductionYLoopAttr)) {
      maxStepFromAttr = attr.getInt();
    } else {
      maxStepFromAttr = kVectorSize;
    }
    return VectorizationMode::ReductionY;
  }

  if (loop->hasAttr(kReductionAllLoopAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kReductionAllLoopAttr);
    if (!attr) {
      return VectorizationMode::None;
    }
    maxStepFromAttr = attr.getInt();
    return VectorizationMode::ReductionX;
  }

  if (loop->hasAttr(kBroadcastLoopAttr)) {
    if (auto attr = loop->getAttrOfType<IntegerAttr>(kBroadcastLoopAttr)) {
      maxStepFromAttr = attr.getInt();
    } else {
      maxStepFromAttr = kVectorSize;
    }
    return VectorizationMode::Broadcast;
  }

  if (loop->hasAttr(kVectorAttr)) {
    auto attr = loop->getAttrOfType<IntegerAttr>(kVectorAttr);
    if (!attr) {
      return VectorizationMode::None;
    }
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
  return (defOp != nullptr) && defOp->getNumOperands() >= kBinaryOpOperandCount &&
         (defOp->getOperand(0) == iterArg || defOp->getOperand(1) == iterArg);
}

/// Maps yield-defining binary arith op to `AtomicRMWKind` when it uses `iterArg` as one combine input.
static std::optional<arith::AtomicRMWKind> matchYieldDefToReductionKind(Operation *defOp, Value iterArg) {
  if (!binaryReductionUsesIterArg(defOp, iterArg)) {
    return std::nullopt;
  }

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
  if (defOp == nullptr) {
    return std::nullopt;
  }

  return matchYieldDefToReductionKind(defOp, loop.getRegionIterArgs()[resultIndex]);
}

static Value createNeutralValue(arith::AtomicRMWKind kind, Type elemType, LoopVectorizationCtx &ctx, Location loc) {
  Attribute neutralAttr = arith::getIdentityValueAttr(kind, elemType, ctx.builder, loc);

  npuvector::NPUVectorType vecType = ctx.getVectorType(elemType);

  Value neutralVec;
  if (ctx.isDynamic()) {
    Value neutralScalar = ctx.builder.create<arith::ConstantOp>(loc, mlir::cast<TypedAttr>(neutralAttr));

    // One (extent, max) pair per result *rank* (not only `?` dims), matching
    // npuvector.transfer_read in `vectorizeLoad` and `allocBroadcastBuffer` when
    // dynamicSizes.size() == npuVecType.getRank().
    SmallVector<Value> dynamicSizes;
    SmallVector<Value> maxSizes;
    for (unsigned i = 0; i < ctx.allVectorSizes.size(); ++i) {
      maxSizes.push_back(ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[i]));
      dynamicSizes.push_back(getRuntimeVectorSizeValue(ctx, i, loc));
    }
    neutralVec = ctx.builder.create<npuvector::BroadcastOp>(loc, vecType, neutralScalar, ValueRange(dynamicSizes),
                                                            ValueRange(maxSizes), ctx.builder.getDenseI64ArrayAttr({}));
  } else {
    auto vecAttr = DenseElementsAttr::get(vecType, neutralAttr);
    neutralVec = ctx.builder.create<arith::ConstantOp>(loc, vecAttr);
  }

  SmallVector<int> dimOrder;
  dimOrder.reserve(ctx.allVectorSizes.size());
  for (unsigned axisIdx = 0; axisIdx < ctx.allVectorSizes.size(); ++axisIdx) {
    dimOrder.push_back(static_cast<int>(axisIdx));
  }
  ctx.valueDimOrder[neutralVec] = std::move(dimOrder);
  return neutralVec;
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

static Value createAffineAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs);

static Value createAffineSub(OpBuilder &builder, Location loc, Value lhs, Value rhs);

static Value createAffineMin(OpBuilder &builder, Location loc, Value lhs, Value rhs);

static bool isReductionXLikeLoop(scf::ForOp loop) {
  return loop->hasAttr(kReductionXLoopAttr) || loop->hasAttr(kReductionAllLoopAttr);
}

static bool isMergedReductionLoop(LoopVectorizationCtx &ctx) {
  return llvm::is_contained(ctx.mergedReductionLoops, ctx.scalarLoop.getOperation());
}

static bool isMergedReductionRoot(LoopVectorizationCtx &ctx) {
  return !ctx.mergedReductionLoops.empty() && ctx.mergedReductionLoops.front() == ctx.scalarLoop.getOperation();
}

static SmallVector<int> buildAxisOrder(unsigned rank) {
  SmallVector<int> axes;
  axes.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    axes.push_back(static_cast<int>(i));
  }
  return axes;
}

static SmallVector<int64_t> buildAllReductionDims(unsigned rank) {
  SmallVector<int64_t> dims;
  dims.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    dims.push_back(static_cast<int64_t>(i));
  }
  return dims;
}

static void markStaticReductionXTailAxisDynamic(LoopVectorizationCtx &ctx) {
  if (ctx.mode != VectorizationMode::ReductionX) {
    return;
  }
  auto dimIter = ctx.allLoopToVectorDim.find(ctx.scalarLoop.getOperation());
  if (dimIter == ctx.allLoopToVectorDim.end()) {
    return;
  }

  const unsigned axisIdx = dimIter->second;
  if (axisIdx >= ctx.allVectorSizes.size() || ctx.allVectorSizeValues[axisIdx]) {
    return;
  }

  std::optional<int64_t> ubOpt = tryConstantIndex(ctx.scalarLoop.getUpperBound());
  std::optional<int64_t> lbOpt = tryConstantIndex(ctx.scalarLoop.getLowerBound());
  if (!ubOpt || !lbOpt || ((*ubOpt - *lbOpt) % ctx.allVectorSizes[axisIdx] == 0)) {
    return;
  }

  ctx.allVectorSizeValues[axisIdx] = ctx.valueMapping.lookupOrDefault(ctx.allMaxStepValues[axisIdx]);
}

struct MergedReductionGroup {
  SmallVector<scf::ForOp> loops;
  SmallVector<int64_t> vectorSizes;
  SmallVector<Value> vectorSizeValues;
  SmallVector<Value> maxStepValues;
  SmallVector<arith::AtomicRMWKind> reductionKinds;
};

static bool isOpInside(Operation *op, Operation *root) {
  for (Operation *cur = op; cur != nullptr; cur = cur->getParentOp()) {
    if (cur == root) {
      return true;
    }
  }
  return false;
}

static bool valueDefinedInsideOp(Value value, Operation *root) {
  if (Operation *defOp = value.getDefiningOp()) {
    return isOpInside(defOp, root);
  }
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    return parentOp != nullptr && isOpInside(parentOp, root);
  }
  return false;
}

static bool hasNestedTaggedLoop(Operation *op) {
  bool found = false;
  op->walk([&](scf::ForOp forOp) {
    if (forOp.getOperation() != op && hasVectorizationAttr(forOp)) {
      found = true;
    }
  });
  return found;
}

static bool collectReductionKindsForGroup(scf::ForOp loop, OpBuilder &builder, MergedReductionGroup &group) {
  if (loop.getNumRegionIterArgs() == 0) {
    return false;
  }
  if (!group.reductionKinds.empty() && group.reductionKinds.size() != loop.getNumRegionIterArgs()) {
    return false;
  }

  SmallVector<arith::AtomicRMWKind> loopKinds;
  loopKinds.reserve(loop.getNumRegionIterArgs());
  for (unsigned idx = 0; idx < loop.getNumRegionIterArgs(); ++idx) {
    auto kind = detectReductionKindForOperand(loop, idx);
    if (!kind || !isNeutralElement(*kind, loop.getInitArgs()[idx], builder)) {
      return false;
    }
    loopKinds.push_back(*kind);
  }

  if (group.reductionKinds.empty()) {
    group.reductionKinds = loopKinds;
    return true;
  }
  return group.reductionKinds == loopKinds;
}

static bool reductionYieldDirectlyUsesChildResult(scf::ForOp loop, scf::ForOp child) {
  auto yieldOp = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yieldOp || loop.getNumRegionIterArgs() != child.getNumResults()) {
    return false;
  }

  for (unsigned idx = 0; idx < loop.getNumRegionIterArgs(); ++idx) {
    Operation *defOp = yieldOp.getOperand(idx).getDefiningOp();
    Value iterArg = loop.getRegionIterArgs()[idx];
    Value childResult = child.getResult(idx);
    if (!binaryReductionUsesIterArg(defOp, iterArg)) {
      return false;
    }
    bool usesChild = defOp->getNumOperands() == kBinaryOpOperandCount &&
                     ((defOp->getOperand(0) == iterArg && defOp->getOperand(1) == childResult) ||
                      (defOp->getOperand(1) == iterArg && defOp->getOperand(0) == childResult));
    if (!usesChild) {
      return false;
    }
  }
  return true;
}

static bool validateLoopForMergedReduction(scf::ForOp loop, scf::ForOp root, OpBuilder &builder,
                                           MergedReductionGroup &group) {
  int64_t maxStep = -1;
  VectorizationMode mode = getVectorizationMode(loop, maxStep);
  if (mode != VectorizationMode::ReductionX || maxStep <= 0 || !isReductionXLikeLoop(loop)) {
    return false;
  }

  auto [skip, isDynamic] = checkLoopEligibility(loop);
  if (skip || !collectReductionKindsForGroup(loop, builder, group)) {
    return false;
  }

  if (isDynamic && loop != root &&
      (valueDefinedInsideOp(loop.getLowerBound(), root.getOperation()) ||
       valueDefinedInsideOp(loop.getUpperBound(), root.getOperation()))) {
    return false;
  }
  return true;
}

static bool appendLoopToMergedReductionGroup(scf::ForOp loop, scf::ForOp root, OpBuilder &builder,
                                             MergedReductionGroup &group) {
  int64_t maxStep = -1;
  VectorizationMode mode = getVectorizationMode(loop, maxStep);
  if (mode != VectorizationMode::ReductionX || maxStep <= 0 || !isReductionXLikeLoop(loop)) {
    return false;
  }

  auto [skip, isDynamic] = checkLoopEligibility(loop);
  if (skip || !collectReductionKindsForGroup(loop, builder, group)) {
    return false;
  }

  int64_t vecSize = computeStaticVectorSize(loop, maxStep);
  Value vecSizeValue = nullptr;

  std::optional<int64_t> ubOpt = tryConstantIndex(loop.getUpperBound());
  std::optional<int64_t> lbOpt = tryConstantIndex(loop.getLowerBound());
  if (isDynamic) {
    if (loop != root && (valueDefinedInsideOp(loop.getLowerBound(), root.getOperation()) ||
                         valueDefinedInsideOp(loop.getUpperBound(), root.getOperation()))) {
      return false;
    }
  }

  Value maxStepValue = builder.create<arith::ConstantIndexOp>(loop.getLoc(), maxStep);
  if (isDynamic) {
    vecSizeValue = computeDynamicVectorSize(loop, maxStepValue, builder, loop.getLoc(), nullptr);
  } else if (ubOpt && lbOpt && ((*ubOpt - *lbOpt) % vecSize != 0)) {
    vecSizeValue = maxStepValue;
  }

  group.loops.push_back(loop);
  group.vectorSizes.push_back(vecSize);
  group.vectorSizeValues.push_back(vecSizeValue);
  group.maxStepValues.push_back(maxStepValue);
  return true;
}

static std::optional<MergedReductionGroup> tryBuildMergedReductionGroup(scf::ForOp root, OpBuilder &builder) {
  if (!isReductionXLikeLoop(root)) {
    return std::nullopt;
  }

  SmallVector<scf::ForOp> loops;
  scf::ForOp current = root;
  while (current) {
    loops.push_back(current);

    scf::ForOp child;
    for (Operation &op : current.getBody()->without_terminator()) {
      auto nestedFor = dyn_cast<scf::ForOp>(&op);
      if (!nestedFor) {
        if (hasNestedTaggedLoop(&op)) {
          return std::nullopt;
        }
        continue;
      }
      if (!hasVectorizationAttr(nestedFor)) {
        if (hasNestedTaggedLoop(nestedFor.getOperation())) {
          return std::nullopt;
        }
        continue;
      }
      if (!isReductionXLikeLoop(nestedFor) || child) {
        return std::nullopt;
      }
      child = nestedFor;
    }
    current = child;
  }

  if (loops.size() < kMinMergedReductionLoopCount) {
    return std::nullopt;
  }
  for (unsigned idx = 0; idx + 1 < loops.size(); ++idx) {
    if (!reductionYieldDirectlyUsesChildResult(loops[idx], loops[idx + 1])) {
      return std::nullopt;
    }
  }

  MergedReductionGroup validateGroup;
  if (std::any_of(loops.begin(), loops.end(), [&](scf::ForOp loop) {
        return !validateLoopForMergedReduction(loop, root, builder, validateGroup);
      })) {
    return std::nullopt;
  }

  MergedReductionGroup group;
  if (std::any_of(loops.begin(), loops.end(),
                  [&](scf::ForOp loop) { return !appendLoopToMergedReductionGroup(loop, root, builder, group); })) {
    return std::nullopt;
  }
  return group;
}

static bool collectReductionNeutralVecs(LoopVectorizationCtx &ctx, Location loc, SmallVectorImpl<Value> &neutralVecs) {
  ctx.reductionKinds.clear();
  ctx.origInits.clear();
  for (unsigned idx = 0; idx < ctx.scalarLoop.getNumRegionIterArgs(); ++idx) {
    auto kind = detectReductionKindForOperand(ctx.scalarLoop, idx);
    if (!kind) {
      return false;
    }

    ctx.reductionKinds.push_back(*kind);
    ctx.origInits.push_back(ctx.scalarLoop.getInitArgs()[idx]);

    Type elemType = ctx.scalarLoop.getRegionIterArgs()[idx].getType();
    Value neutralVec = createNeutralValue(*kind, elemType, ctx, loc);
    if (!neutralVec) {
      return false;
    }
    neutralVecs.push_back(neutralVec);
  }
  return true;
}

static scf::ForOp createEmptyVectorizedLoop(LoopVectorizationCtx &ctx) {
  Location loc = ctx.scalarLoop.getLoc();
  markStaticReductionXTailAxisDynamic(ctx);

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
      if (!collectReductionNeutralVecs(ctx, loc, neutralVecs)) {
        return nullptr;
      }
    }
  }

  Value upperBound = ctx.scalarLoop.getUpperBound();
  Value lowerBound = ctx.scalarLoop.getLowerBound();

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
  if (auto constantIndexOp = indexValue.getDefiningOp<arith::ConstantIndexOp>()) {
    return constantIndexOp.value();
  }
  if (auto constantOp = indexValue.getDefiningOp<arith::ConstantOp>()) {
    if (!indexValue.getType().isIndex()) {
      return std::nullopt;
    }
    if (auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      return integerAttr.getValue().getSExtValue();
    }
  }
  return std::nullopt;
}

static Value createAffineBinaryApply(OpBuilder &builder, Location loc, Value lhs, Value rhs, AffineExpr expr) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeTwo> operands{lhs, rhs};
  AffineMap map = AffineMap::get(kSmallVectorSizeTwo, kSmallVectorSizeZero, expr, context);
  return builder.create<affine::AffineApplyOp>(loc, map, operands).getResult();
}

static Value createAffineAdd(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  MLIRContext *context = builder.getContext();
  AffineExpr expr = getAffineDimExpr(0, context) + getAffineDimExpr(1, context);
  return createAffineBinaryApply(builder, loc, lhs, rhs, expr);
}

static Value createAffineSub(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  MLIRContext *context = builder.getContext();
  AffineExpr expr = getAffineDimExpr(0, context) - getAffineDimExpr(1, context);
  return createAffineBinaryApply(builder, loc, lhs, rhs, expr);
}

static Value createAffineMinWithConstant(OpBuilder &builder, Location loc, Value value, int64_t constant) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeOne> operands{value};
  SmallVector<AffineExpr, kSmallVectorSizeTwo> minExprs{getAffineDimExpr(0, context),
                                                        getAffineConstantExpr(constant, context)};
  AffineMap minMap = AffineMap::get(kSmallVectorSizeOne, kSmallVectorSizeZero, minExprs, context);
  return builder.create<affine::AffineMinOp>(loc, minMap, operands).getResult();
}

static Value createAffineMin(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeTwo> operands{lhs, rhs};
  SmallVector<AffineExpr, kSmallVectorSizeTwo> minExprs{getAffineDimExpr(0, context), getAffineDimExpr(1, context)};
  AffineMap minMap = AffineMap::get(kSmallVectorSizeTwo, kSmallVectorSizeZero, minExprs, context);
  return builder.create<affine::AffineMinOp>(loc, minMap, operands).getResult();
}

static Value createAffineMax(OpBuilder &builder, Location loc, Value lhs, Value rhs) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeTwo> operands{lhs, rhs};
  SmallVector<AffineExpr, kSmallVectorSizeTwo> maxExprs{getAffineDimExpr(0, context), getAffineDimExpr(1, context)};
  AffineMap maxMap = AffineMap::get(kSmallVectorSizeTwo, kSmallVectorSizeZero, maxExprs, context);
  return builder.create<affine::AffineMaxOp>(loc, maxMap, operands).getResult();
}

static Value createAffineMaxWithConstant(OpBuilder &builder, Location loc, Value value, int64_t constant) {
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeOne> operands{value};
  SmallVector<AffineExpr, kSmallVectorSizeTwo> maxExprs{getAffineDimExpr(0, context),
                                                        getAffineConstantExpr(constant, context)};
  AffineMap maxMap = AffineMap::get(kSmallVectorSizeOne, kSmallVectorSizeZero, maxExprs, context);
  return builder.create<affine::AffineMaxOp>(loc, maxMap, operands).getResult();
}

static Value getRuntimeVectorSizeValue(LoopVectorizationCtx &ctx, unsigned axisIdx, Location loc) {
  if (axisIdx < ctx.currentVectorSizeValues.size() && ctx.currentVectorSizeValues[axisIdx]) {
    return ctx.valueMapping.lookupOrDefault(ctx.currentVectorSizeValues[axisIdx]);
  }
  if (axisIdx < ctx.allVectorSizeValues.size() && ctx.allVectorSizeValues[axisIdx]) {
    return ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[axisIdx]);
  }
  return ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[axisIdx]);
}

static bool gatherVectorDynExtents(Value vectorVal, SmallVectorImpl<Value> &outExtents) {
  Operation *defOp = vectorVal.getDefiningOp();
  if (defOp == nullptr) {
    return false;
  }

  if (auto readOp = mlir::dyn_cast<npuvector::TransferReadOp>(defOp)) {
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vectorVal.getType());
    if (!nvt) {
      return false;
    }
    npuvector::TransferReadOp::Adaptor readAdaptor(readOp);
    ValueRange dynSizes = readAdaptor.getDynamicSizes();
    if (static_cast<int64_t>(dynSizes.size()) != static_cast<int64_t>(nvt.getRank())) {
      return false;
    }
    outExtents.assign(dynSizes.begin(), dynSizes.end());
    return true;
  }

  if (auto transposeOp = mlir::dyn_cast<npuvector::TransposeOp>(defOp)) {
    SmallVector<Value> innerExtents;
    if (!gatherVectorDynExtents(transposeOp.getVector(), innerExtents)) {
      return false;
    }
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    auto resultNvt = mlir::dyn_cast<npuvector::NPUVectorType>(vectorVal.getType());
    if (!resultNvt || perm.size() != static_cast<size_t>(resultNvt.getRank())) {
      return false;
    }
    const auto rank = static_cast<unsigned>(resultNvt.getRank());
    outExtents.resize(rank);
    for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
      const int64_t srcDim = perm[resultDim];
      if (srcDim < 0 || static_cast<size_t>(srcDim) >= innerExtents.size()) {
        return false;
      }
      outExtents[resultDim] = innerExtents[static_cast<unsigned>(srcDim)];
    }
    return true;
  }

  if (auto sliceOp = mlir::dyn_cast<npuvector::ExtractSliceOp>(defOp)) {
    ArrayRef<int64_t> keepDims = sliceOp.getKeepDims();
    ValueRange sizes = sliceOp.getSizes();
    outExtents.clear();
    outExtents.reserve(keepDims.size());
    for (int64_t d : keepDims) {
      if (d < 0 || static_cast<size_t>(d) >= sizes.size()) {
        return false;
      }
      outExtents.push_back(sizes[static_cast<unsigned>(d)]);
    }
    return true;
  }

  return false;
}

static std::optional<unsigned> matchExtentValueToCtxAxis(Value extent, const LoopVectorizationCtx &ctx) {
  Value mappedExt = ctx.valueMapping.lookupOrDefault(extent);
  llvm::SmallVector<unsigned, kSmallVectorSizeFour> hits;
  for (unsigned axisIdx = 0; axisIdx < ctx.allVectorSizes.size(); ++axisIdx) {
    if (ctx.allVectorSizeValues[axisIdx]) {
      Value refLen = ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[axisIdx]);
      if (mappedExt == refLen || extent == refLen) {
        hits.push_back(axisIdx);
      }
    } else {
      const std::optional<int64_t> extentAsConstant = tryConstantIndex(mappedExt);
      if (extentAsConstant && static_cast<int64_t>(ctx.allVectorSizes[axisIdx]) == *extentAsConstant) {
        hits.push_back(axisIdx);
      }
    }
  }
  if (hits.size() != 1) {
    return std::nullopt;
  }
  return hits.front();
}

static void reconcileValueDimOrderWithTileExtents(Value loadedVector, SmallVector<int> &sortedDims,
                                                  const LoopVectorizationCtx &ctx) {
  SmallVector<Value> perDimExtents;
  if (!gatherVectorDynExtents(loadedVector, perDimExtents)) {
    return;
  }
  if (perDimExtents.size() != sortedDims.size()) {
    return;
  }

  SmallVector<int> fromExtents;
  fromExtents.reserve(perDimExtents.size());
  llvm::SmallDenseSet<unsigned> usedAxes;
  for (Value ext : perDimExtents) {
    std::optional<unsigned> axisOpt = matchExtentValueToCtxAxis(ext, ctx);
    if (!axisOpt || !usedAxes.insert(*axisOpt).second) {
      return;
    }
    fromExtents.push_back(static_cast<int>(*axisOpt));
  }

  if (sortedDims.size() == fromExtents.size() &&
      std::equal(sortedDims.begin(), sortedDims.end(), fromExtents.begin())) {
    return;
  }
  sortedDims = std::move(fromExtents);
}

static bool axisHasCurrentTileSize(const LoopVectorizationCtx &ctx, int axis) {
  return axis >= 0 && static_cast<unsigned>(axis) < ctx.currentVectorSizeValues.size() &&
         ctx.currentVectorSizeValues[static_cast<unsigned>(axis)] != nullptr;
}

static std::optional<unsigned> getCurrentReductionTileAxis(LoopVectorizationCtx &ctx) {
  if (ctx.mode != VectorizationMode::ReductionX) {
    return std::nullopt;
  }
  auto dimIter = ctx.allLoopToVectorDim.find(ctx.scalarLoop.getOperation());
  if (dimIter == ctx.allLoopToVectorDim.end()) {
    return std::nullopt;
  }
  unsigned axis = dimIter->second;
  if (!axisHasCurrentTileSize(ctx, static_cast<int>(axis))) {
    return std::nullopt;
  }
  return axis;
}

static Value createCurrentTileSlice(Value vecArg, LoopVectorizationCtx &ctx, ArrayRef<int> dimToAxis, unsigned tileAxis,
                                    CurrentTileSliceInfo *sliceInfo = nullptr) {
  auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vecArg.getType());
  const unsigned rank = nvt ? static_cast<unsigned>(nvt.getRank()) : 0;
  if (!nvt || rank == 0 || dimToAxis.size() != rank || !llvm::is_contained(dimToAxis, static_cast<int>(tileAxis))) {
    return vecArg;
  }

  Location loc = vecArg.getLoc();
  SmallVector<Value> sizes;
  SmallVector<int64_t> keepDims;
  SmallVector<int64_t> resultShape(nvt.getShape().begin(), nvt.getShape().end());
  sizes.reserve(rank);
  keepDims.reserve(rank);

  for (auto [dim, axis] : llvm::enumerate(dimToAxis)) {
    if (axis < 0) {
      return vecArg;
    }
    Value size = getRuntimeVectorSizeValue(ctx, static_cast<unsigned>(axis), loc);
    if (!size) {
      return vecArg;
    }
    sizes.push_back(size);
    if (static_cast<unsigned>(axis) == tileAxis) {
      resultShape[dim] = ShapedType::kDynamic;
    }
    keepDims.push_back(static_cast<int64_t>(dim));
  }

  Value c0 = ctx.builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = ctx.builder.create<arith::ConstantIndexOp>(loc, 1);
  // Slice the accumulator by vector lane. The memory-space offset is already carried by transfer_read's IV.
  SmallVector<Value> offsets(rank, c0);
  SmallVector<Value> strides(rank, c1);

  auto sliceType = npuvector::NPUVectorType::get(resultShape, nvt.getElementType());
  Value slice =
    ctx.builder.create<npuvector::ExtractSliceOp>(loc, sliceType, vecArg, ValueRange(offsets), ValueRange(sizes),
                                                  ValueRange(strides), ctx.builder.getDenseI64ArrayAttr(keepDims));
  ctx.valueDimOrder[slice] = SmallVector<int>(dimToAxis.begin(), dimToAxis.end());
  if (sliceInfo != nullptr) {
    sliceInfo->fullVector = vecArg;
    sliceInfo->offsets = offsets;
    sliceInfo->sizes = sizes;
    sliceInfo->strides = strides;
    sliceInfo->dimToAxis.assign(dimToAxis.begin(), dimToAxis.end());
  }
  return slice;
}

static Value insertCurrentTileSlice(Value tileValue, LoopVectorizationCtx &ctx, const CurrentTileSliceInfo &sliceInfo,
                                    Location loc) {
  auto destType = mlir::dyn_cast<npuvector::NPUVectorType>(sliceInfo.fullVector.getType());
  if (!destType) {
    return tileValue;
  }
  Value fullValue = ctx.builder.create<npuvector::InsertSliceOp>(
    loc, destType, tileValue, sliceInfo.fullVector, ValueRange(sliceInfo.offsets), ValueRange(sliceInfo.sizes),
    ValueRange(sliceInfo.strides));
  ctx.valueDimOrder[fullValue] = sliceInfo.dimToAxis;
  return fullValue;
}

static void collectVectorAxes(const LoopVectorizationCtx &ctx, SmallVectorImpl<std::pair<Value, int>> &axes) {
  axes.clear();
  DenseSet<Value> seen;
  for (auto &[op, dim] : ctx.allLoopToVectorDim) {
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp || !seen.insert(forOp.getInductionVar()).second) {
      continue;
    }
    axes.push_back({forOp.getInductionVar(), static_cast<int>(dim)});
  }
  if (ctx.vectorizationAxis && ctx.vectorAxisVectorDim && seen.insert(ctx.vectorizationAxis).second) {
    axes.push_back({ctx.vectorizationAxis, static_cast<int>(*ctx.vectorAxisVectorDim)});
  }
  llvm::sort(axes, [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
}

static bool affineExprCoeffForIndex(AffineExpr expr, ArrayRef<int64_t> operandCoeffs, unsigned numDims,
                                    int64_t &coeff) {
  coeff = 0;
  if (isa<AffineConstantExpr>(expr)) {
    return true;
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
    unsigned pos = dimExpr.getPosition();
    if (pos >= operandCoeffs.size()) {
      return false;
    }
    coeff = operandCoeffs[pos];
    return true;
  }
  if (auto symbolExpr = dyn_cast<AffineSymbolExpr>(expr)) {
    unsigned pos = numDims + symbolExpr.getPosition();
    if (pos >= operandCoeffs.size()) {
      return false;
    }
    coeff = operandCoeffs[pos];
    return true;
  }

  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binaryExpr) {
    return false;
  }

  int64_t lhsCoeff = 0;
  int64_t rhsCoeff = 0;
  if (!affineExprCoeffForIndex(binaryExpr.getLHS(), operandCoeffs, numDims, lhsCoeff) ||
      !affineExprCoeffForIndex(binaryExpr.getRHS(), operandCoeffs, numDims, rhsCoeff)) {
    return false;
  }

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
  if (definingOp == nullptr) {
    return true;
  }

  if (auto addOp = dyn_cast<arith::AddIOp>(definingOp)) {
    int64_t lhsCoeff = 0;
    int64_t rhsCoeff = 0;
    if (!indexCoeffForVectorAxis(addOp.getLhs(), vectorAxis, lhsCoeff) ||
        !indexCoeffForVectorAxis(addOp.getRhs(), vectorAxis, rhsCoeff)) {
      return false;
    }
    coeff = lhsCoeff + rhsCoeff;
    return true;
  }
  if (auto subOp = dyn_cast<arith::SubIOp>(definingOp)) {
    int64_t lhsCoeff = 0;
    int64_t rhsCoeff = 0;
    if (!indexCoeffForVectorAxis(subOp.getLhs(), vectorAxis, lhsCoeff) ||
        !indexCoeffForVectorAxis(subOp.getRhs(), vectorAxis, rhsCoeff)) {
      return false;
    }
    coeff = lhsCoeff - rhsCoeff;
    return true;
  }
  if (auto affineApplyOp = dyn_cast<affine::AffineApplyOp>(definingOp)) {
    SmallVector<int64_t> operandCoeffs;
    operandCoeffs.reserve(affineApplyOp.getMapOperands().size());
    for (Value operand : affineApplyOp.getMapOperands()) {
      int64_t operandCoeff = 0;
      if (!indexCoeffForVectorAxis(operand, vectorAxis, operandCoeff)) {
        return false;
      }
      operandCoeffs.push_back(operandCoeff);
    }
    return affineExprCoeffForIndex(affineApplyOp.getAffineMap().getResult(0), operandCoeffs,
                                   affineApplyOp.getAffineMap().getNumDims(), coeff);
  }

  return !definitionGraphContainsValue(index, vectorAxis);
}

static int getVectorDimForIndex(Value index, const LoopVectorizationCtx &ctx) {
  int directDim = ctx.getVectorDimForIV(index);
  if (directDim >= 0) {
    return directDim;
  }

  SmallVector<std::pair<Value, int>> axes;
  collectVectorAxes(ctx, axes);
  int matchedDim = -1;
  for (auto [axis, dim] : axes) {
    int64_t coeff = 0;
    if (!indexCoeffForVectorAxis(index, axis, coeff)) {
      return -1;
    }
    if (coeff == 0) {
      continue;
    }
    if (coeff != 1 || matchedDim >= 0) {
      return -1;
    }
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
    if (dim >= 0) {
      activeInAppearOrder.push_back(dim);
    }
  }
}

static LogicalResult collectAccessAxisToMemDim(Operation *op, ValueRange indices, const LoopVectorizationCtx &ctx,
                                               StringRef accessName, SmallVectorImpl<int64_t> &axisToMemDim) {
  axisToMemDim.assign(ctx.allVectorSizes.size(), -1);
  for (auto [memDim, idx] : llvm::enumerate(indices)) {
    int axis = getVectorDimForIndex(idx, ctx);
    if (axis < 0) {
      continue;
    }
    if (static_cast<size_t>(axis) >= axisToMemDim.size()) {
      op->emitError("npuvector-vectorize: ") << accessName << " index maps to invalid vector dim";
      return failure();
    }
    int64_t &mappedMemDim = axisToMemDim[static_cast<size_t>(axis)];
    if (mappedMemDim != -1) {
      op->emitError("npuvector-vectorize: one vector axis maps to multiple ") << accessName << " dimensions";
      return failure();
    }
    mappedMemDim = static_cast<int64_t>(memDim);
  }
  return success();
}

struct PermMapFromAxisOrderParams {
  Operation *op = nullptr;
  ArrayRef<int> axisOrder;
  ArrayRef<int64_t> axisToMemDim;
  unsigned memRefRank = 0;
  MLIRContext *context = nullptr;
  StringRef opName;
};

static FailureOr<AffineMap> buildPermutationMapFromAxisOrder(const PermMapFromAxisOrderParams &params) {
  SmallVector<AffineExpr> results;
  results.reserve(params.axisOrder.size());
  for (int axis : params.axisOrder) {
    if (axis < 0 || static_cast<size_t>(axis) >= params.axisToMemDim.size() ||
        params.axisToMemDim[static_cast<size_t>(axis)] < 0) {
      params.op->emitError("npuvector-vectorize: cannot infer ")
        << params.opName << " permutation_map from access indices";
      return failure();
    }
    results.push_back(
      getAffineDimExpr(static_cast<unsigned>(params.axisToMemDim[static_cast<size_t>(axis)]), params.context));
  }
  return AffineMap::get(params.memRefRank, kSmallVectorSizeZero, results, params.context);
}

static FailureOr<AffineMap> buildTransferReadPermutationMap(memref::LoadOp loadOp, LoopVectorizationCtx &ctx,
                                                            ArrayRef<int> activeInAppearOrder) {
  SmallVector<int64_t> axisToMemDim;
  if (failed(collectAccessAxisToMemDim(loadOp.getOperation(), loadOp.getIndices(), ctx, "load", axisToMemDim))) {
    return failure();
  }
  return buildPermutationMapFromAxisOrder(PermMapFromAxisOrderParams{
    loadOp.getOperation(), activeInAppearOrder, axisToMemDim, static_cast<unsigned>(loadOp.getIndices().size()),
    ctx.builder.getContext(), "transfer_read"});
}

static bool loadIndicesAreLoopInvariant(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
  if (ctx.vf1FuncLevelNoAnchor) {
    return false;
  }
  if (ctx.scalarLoop.getOperation() == nullptr) {
    return false;
  }
  Value vecAxis = ctx.getVectorizationAxis();
  Value mappedVecAxis = ctx.valueMapping.lookupOrDefault(vecAxis);
  for (Value idx : loadOp.getIndices()) {
    if (getVectorDimForIndex(idx, ctx) >= 0) {
      return false;
    }
    if (idx == vecAxis || idx == mappedVecAxis) {
      return false;
    }
    if (idx.getParentBlock() == ctx.scalarLoop.getBody()) {
      return false;
    }
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

  npuvector::NPUVectorType readVecType = npuvector::NPUVectorType::get({}, elemType);
  Value padding = ctx.builder.create<arith::ConstantOp>(loc, ctx.builder.getZeroAttr(elemType));
  auto transferRead = ctx.builder.create<npuvector::TransferReadOp>(loc, readVecType, mappedMemRef, ValueRange(indices),
                                                                    padding, Value(), ValueRange(), ValueRange());
  Value result = transferRead.getResult();
  ctx.valueDimOrder[result] = SmallVector<int>();
  return result;
}

/// Map sorted global dims back to appear-order indices for npuvector.transpose.
static SmallVector<int64_t> buildAppearOrderTransposePerm(ArrayRef<int> activeInAppearOrder, ArrayRef<int> sortedDims) {
  const unsigned subRank = activeInAppearOrder.size();
  SmallVector<int64_t> transposePerm(subRank);
  for (unsigned i = 0; i < subRank; ++i) {
    int targetGlobal = sortedDims[i];
    unsigned j = 0;
    for (; j < subRank; ++j) {
      if (activeInAppearOrder[j] == targetGlobal) {
        break;
      }
    }
    transposePerm[i] = static_cast<int64_t>(j);
  }
  return transposePerm;
}

/// valueDimOrder entry: global ctx axis id per result dimension.
static SmallVector<int> buildAxisPerResultDim(ArrayRef<int> activeInAppearOrder, ArrayRef<int64_t> transposePerm,
                                              bool analyzedTranspose) {
  const unsigned subRank = activeInAppearOrder.size();
  SmallVector<int> axisPerResultDim(subRank);
  if (analyzedTranspose) {
    for (unsigned r = 0; r < subRank; ++r) {
      axisPerResultDim[r] = activeInAppearOrder[static_cast<unsigned>(transposePerm[r])];
    }
    return axisPerResultDim;
  }
  for (unsigned k = 0; k < subRank; ++k) {
    axisPerResultDim[k] = activeInAppearOrder[k];
  }
  return axisPerResultDim;
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
  if (subRank == 0) {
    return vectorizeLoadSubRankZero(loadOp, ctx, mappedMemRef, elemType, loc);
  }

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
    dynamicSizes.push_back(getRuntimeVectorSizeValue(ctx, static_cast<unsigned>(gid), loc));
  }

  FailureOr<AffineMap> permutationMap = buildTransferReadPermutationMap(loadOp, ctx, activeInAppearOrder);
  if (failed(permutationMap)) {
    return {};
  }
  AffineMap defaultMap =
    AffineMap::getMinorIdentityMap(static_cast<unsigned>(indices.size()), subRank, ctx.builder.getContext());
  auto transferRead =
    (*permutationMap == defaultMap)
      ? ctx.builder.create<npuvector::TransferReadOp>(loc, readVecType, mappedMemRef, ValueRange(indices), padding,
                                                      Value(), ValueRange(dynamicSizes), ValueRange(maxSizes))
      : ctx.builder.create<npuvector::TransferReadOp>(loc, readVecType, mappedMemRef, ValueRange(indices), padding,
                                                      Value(), ValueRange(dynamicSizes), ValueRange(maxSizes),
                                                      *permutationMap);

  Value result = transferRead.getResult();
  SmallVector<int64_t> transposePerm;
  if (analyzedTranspose) {
    transposePerm = buildAppearOrderTransposePerm(activeInAppearOrder, sortedDims);
    result = ctx.builder.create<npuvector::TransposeOp>(loc, result, transposePerm);
  }

  // valueDimOrder[result][r] = global ctx axis id for **result** dimension r. transfer_read dim k
  // follows memref index order -> axis activeInAppearOrder[k]. npuvector.transpose: result dim r
  // comes from operand dim transposePerm[r] (see gatherVectorDynExtents on TransposeOp).
  SmallVector<int> axisPerResultDim = buildAxisPerResultDim(activeInAppearOrder, transposePerm, analyzedTranspose);
  reconcileValueDimOrderWithTileExtents(result, axisPerResultDim, ctx);
  ctx.valueDimOrder[result] = axisPerResultDim;

  return result;
}

static LogicalResult buildStoreTargetTypeForDimOrder(memref::StoreOp storeOp, ArrayRef<int> storeDimOrder,
                                                     LoopVectorizationCtx &ctx, npuvector::NPUVectorType &outTy,
                                                     SmallVectorImpl<int> &outResultDimToCtxAxis) {
  const auto ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  Type elemType = storeOp.getMemRefType().getElementType();
  SmallVector<int64_t> brShape;
  brShape.reserve(storeDimOrder.size());
  for (int ax : storeDimOrder) {
    if (ax < 0 || static_cast<unsigned>(ax) >= ctxRank) {
      return failure();
    }
    const auto uax = static_cast<unsigned>(ax);
    brShape.push_back(ctx.allVectorSizeValues[uax] ? ShapedType::kDynamic : ctx.allVectorSizes[uax]);
  }
  outTy = npuvector::NPUVectorType::get(brShape, elemType);
  outResultDimToCtxAxis.assign(storeDimOrder.begin(), storeDimOrder.end());
  return success();
}

static bool ensureBroadcastStoreValue(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, ArrayRef<int> storeDimOrder,
                                      Value storeValue, Value &vectorValue) {
  if (vectorValue && mlir::isa<npuvector::NPUVectorType>(vectorValue.getType())) {
    return true;
  }
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

struct ValidateMultiDimStoreValueAxesParams {
  mutable memref::StoreOp storeOp;
  LoopVectorizationCtx &ctx;
  Value vectorValue;
  int64_t vecRank = 0;
  ArrayRef<int> storeDimOrder;
  SmallVector<int> &valueDimOrdOut;
};

static bool validateMultiDimStoreValueAxes(const ValidateMultiDimStoreValueAxesParams &params) {
  const bool hadDimOrderForValue = params.ctx.valueDimOrder.count(params.vectorValue) != 0;
  if (hadDimOrderForValue) {
    params.valueDimOrdOut = params.ctx.valueDimOrder[params.vectorValue];
  }
  if (!hadDimOrderForValue || params.valueDimOrdOut.empty()) {
    params.storeOp.emitError(
      "npuvector-vectorize: multi-index / multi-rank store requires non-empty "
      "valueDimOrder on the stored vector value");
    return false;
  }
  if (static_cast<int64_t>(params.valueDimOrdOut.size()) != params.vecRank) {
    params.storeOp.emitError("npuvector-vectorize: valueDimOrder rank does not match stored npuvector rank");
    return false;
  }
  for (int axisIdx : params.valueDimOrdOut) {
    if (!llvm::is_contained(params.storeDimOrder, axisIdx)) {
      params.storeOp.emitError(
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
  for (int axisIdx : valueDimOrd) {
    valueAxisSet.insert(axisIdx);
  }
  llvm::DenseSet<int> seenIntersectAxis;
  intersectOut.clear();
  intersectOut.reserve(valueDimOrd.size());
  for (int axisIdx : storeDimOrder) {
    if (valueAxisSet.find(axisIdx) == valueAxisSet.end()) {
      continue;
    }
    if (!seenIntersectAxis.insert(axisIdx).second) {
      continue;
    }
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

struct StoreVectorAxisAdjustParams {
  mutable memref::StoreOp storeOp;
  LoopVectorizationCtx &ctx;
  Location loc;
  ArrayRef<int> valueDimOrd;
  ArrayRef<int> storeDimOrder;
  ArrayRef<int> intersectStoreDimOrder;
  Value &vectorValue;
};

static bool transposeStoreVectorIfAxesDiffer(const StoreVectorAxisAdjustParams &params) {
  if (params.valueDimOrd == params.intersectStoreDimOrder) {
    return true;
  }
  SmallVector<int64_t> perm(params.intersectStoreDimOrder.size());
  for (unsigned rowIdx = 0; rowIdx < params.intersectStoreDimOrder.size(); ++rowIdx) {
    bool matched = false;
    for (unsigned j = 0; j < params.valueDimOrd.size(); ++j) {
      if (params.valueDimOrd[j] == params.intersectStoreDimOrder[rowIdx]) {
        perm[rowIdx] = static_cast<int64_t>(j);
        matched = true;
        break;
      }
    }
    if (!matched) {
      params.storeOp.emitError(
        "npuvector-vectorize: cannot map intersected store axis order to valueDimOrder for "
        "transpose");
      return false;
    }
  }
  params.vectorValue = params.ctx.builder.create<npuvector::TransposeOp>(params.loc, params.vectorValue, perm);
  params.ctx.valueDimOrder[params.vectorValue] =
    SmallVector<int>(params.intersectStoreDimOrder.begin(), params.intersectStoreDimOrder.end());
  return true;
}

static bool rankLiftStoreVectorIfExtraIndices(const StoreVectorAxisAdjustParams &params) {
  if (params.storeDimOrder.size() <= params.intersectStoreDimOrder.size()) {
    return true;
  }
  npuvector::NPUVectorType targetTy;
  SmallVector<int> resultDimToCtxAxis;
  if (failed(buildStoreTargetTypeForDimOrder(params.storeOp, params.storeDimOrder, params.ctx, targetTy,
                                             resultDimToCtxAxis))) {
    params.storeOp.emitError("npuvector-vectorize: store index maps to invalid vector dim");
    return false;
  }
  Value lifted = vectorizeBroadcastScalar(params.vectorValue, params.ctx, targetTy, resultDimToCtxAxis);
  if (lifted == nullptr) {
    params.storeOp.emitError(
      "npuvector-vectorize: store rank-lift broadcast failed (see diagnostic "
      "on npuvector.broadcast / valueDimOrder)");
    return false;
  }
  params.vectorValue = lifted;
  return true;
}

static bool reorderStoreVectorForIndices(memref::StoreOp storeOp, LoopVectorizationCtx &ctx, Location loc,
                                         ArrayRef<int> storeDimOrder, Value &vectorValue) {
  auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!npuVecType) {
    return true;
  }
  const int64_t vecRank = npuVecType.getRank();
  if (vecRank <= 1 && storeDimOrder.size() <= 1) {
    return true;
  }

  SmallVector<int> valueDimOrd;
  if (!validateMultiDimStoreValueAxes(
        ValidateMultiDimStoreValueAxesParams{storeOp, ctx, vectorValue, vecRank, storeDimOrder, valueDimOrd})) {
    return false;
  }
  SmallVector<int> intersectStoreDimOrder;
  if (!intersectStoreAxesWithValueAxes(storeOp, storeDimOrder, valueDimOrd, intersectStoreDimOrder)) {
    return false;
  }
  StoreVectorAxisAdjustParams adjustParams{storeOp,    ctx, loc, valueDimOrd, storeDimOrder, intersectStoreDimOrder,
                                           vectorValue};
  if (!transposeStoreVectorIfAxesDiffer(adjustParams)) {
    return false;
  }
  return rankLiftStoreVectorIfExtraIndices(adjustParams);
}

static SmallVector<int> collectStoreDimOrder(memref::StoreOp storeOp, const LoopVectorizationCtx &ctx) {
  SmallVector<int> storeDimOrder;
  for (Value idx : storeOp.getIndices()) {
    int dim = getVectorDimForIndex(idx, ctx);
    if (dim >= 0) {
      storeDimOrder.push_back(dim);
    }
  }
  return storeDimOrder;
}

static FailureOr<AffineMap> buildTransferWritePermutationMap(memref::StoreOp storeOp, LoopVectorizationCtx &ctx,
                                                             Value vectorValue, ArrayRef<int> storeDimOrder) {
  auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!npuVecType) {
    storeOp.emitError("npuvector-vectorize: transfer_write value must be NPUVectorType");
    return failure();
  }

  const int64_t vecRank = npuVecType.getRank();
  int64_t memRefRank = static_cast<int64_t>(storeOp.getIndices().size());
  if (vecRank == 0) {
    return AffineMap::get(static_cast<unsigned>(memRefRank), kSmallVectorSizeZero, ctx.builder.getContext());
  }

  SmallVector<int64_t> axisToMemDim;
  if (failed(collectAccessAxisToMemDim(storeOp.getOperation(), storeOp.getIndices(), ctx, "store", axisToMemDim))) {
    return failure();
  }

  SmallVector<int> valueDimOrder;
  auto dimOrderIt = ctx.valueDimOrder.find(vectorValue);
  if (dimOrderIt != ctx.valueDimOrder.end()) {
    valueDimOrder = dimOrderIt->second;
  } else if (vecRank == 1 && storeDimOrder.size() == 1) {
    valueDimOrder.assign(storeDimOrder.begin(), storeDimOrder.end());
  }

  if (static_cast<int64_t>(valueDimOrder.size()) != vecRank) {
    storeOp.emitError("npuvector-vectorize: cannot infer transfer_write permutation_map from valueDimOrder");
    return failure();
  }

  return buildPermutationMapFromAxisOrder(PermMapFromAxisOrderParams{storeOp.getOperation(), valueDimOrder,
                                                                     axisToMemDim, static_cast<unsigned>(memRefRank),
                                                                     ctx.builder.getContext(), "transfer_write"});
}

// limit.
struct CreateTransferWriteParams {
  mutable memref::StoreOp storeOp;
  LoopVectorizationCtx &ctx;
  Location loc;
  Value vectorValue;
  Value mappedMemRef;
  ValueRange indices;
  ArrayRef<int> storeDimOrder;
};

static bool createTransferWriteWithPermutationMap(const CreateTransferWriteParams &params) {
  FailureOr<AffineMap> permutationMap =
    buildTransferWritePermutationMap(params.storeOp, params.ctx, params.vectorValue, params.storeDimOrder);
  if (failed(permutationMap)) {
    return false;
  }
  auto npuVecType = mlir::cast<npuvector::NPUVectorType>(params.vectorValue.getType());
  AffineMap defaultMap =
    AffineMap::getMinorIdentityMap(static_cast<unsigned>(params.indices.size()),
                                   static_cast<unsigned>(npuVecType.getRank()), params.ctx.builder.getContext());
  if (*permutationMap == defaultMap) {
    params.ctx.builder.create<npuvector::TransferWriteOp>(params.loc, Type(), params.vectorValue, params.mappedMemRef,
                                                          params.indices);
  } else {
    params.ctx.builder.create<npuvector::TransferWriteOp>(params.loc, Type(), params.vectorValue, params.mappedMemRef,
                                                          params.indices, *permutationMap);
  }
  return true;
}

static void vectorizeStore(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  Location loc = storeOp.getLoc();

  SmallVector<int> storeDimOrder = collectStoreDimOrder(storeOp, ctx);

  Value storeValue = storeOp.getValue();
  Value vectorValue = ctx.valueMapping.lookupOrNull(storeValue);
  if (!ensureBroadcastStoreValue(storeOp, ctx, storeDimOrder, storeValue, vectorValue)) {
    return;
  }

  SmallVector<Value> indices;
  indices.reserve(storeOp.getIndices().size());
  std::transform(storeOp.getIndices().begin(), storeOp.getIndices().end(), std::back_inserter(indices),
                 [&ctx](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });
  if (!mlir::isa<npuvector::NPUVectorType>(vectorValue.getType())) {
    llvm_unreachable("vectorizeStore: vector value must be NPUVectorType");
  }

  if (!reorderStoreVectorForIndices(storeOp, ctx, loc, storeDimOrder, vectorValue)) {
    return;
  }

  Value memRef = storeOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);
  if (!createTransferWriteWithPermutationMap(
        CreateTransferWriteParams{storeOp, ctx, loc, vectorValue, mappedMemRef, ValueRange(indices), storeDimOrder})) {
    return;
  }
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
    if (!nvt) {
      continue;
    }
    auto r = static_cast<int64_t>(nvt.getRank());
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

struct LookupPeerBroadcastAxesParams {
  Operation *arithOp = nullptr;
  LoopVectorizationCtx &ctx;
  const SmallVectorImpl<Value> &vecOperands;
  llvm::ArrayRef<unsigned> scalarIndices;
  npuvector::NPUVectorType &refVecType;
  SmallVectorImpl<int> &outAxes;
};

static bool lookupPeerBroadcastAxes(const LookupPeerBroadcastAxesParams &params) {
  params.outAxes.clear();
  const unsigned refRank = static_cast<unsigned>(params.refVecType.getRank());
  if (refRank == 0) {
    return true;
  }

  llvm::SmallDenseSet<int> unionAxes;
  bool hasCompleteOrders = true;
  for (Value operand : params.vecOperands) {
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(operand.getType());
    if (!nvt) {
      continue;
    }
    auto it = params.ctx.valueDimOrder.find(operand);
    if (it == params.ctx.valueDimOrder.end() || it->second.size() != static_cast<unsigned>(nvt.getRank())) {
      hasCompleteOrders = false;
      break;
    }
    for (int axis : it->second) {
      if (axis < 0 || static_cast<unsigned>(axis) >= params.ctx.allVectorSizes.size()) {
        return false;
      }
      unionAxes.insert(axis);
    }
  }
  if (hasCompleteOrders && unionAxes.size() > refRank) {
    params.outAxes.assign(unionAxes.begin(), unionAxes.end());
    llvm::sort(params.outAxes);
    SmallVector<int64_t> shape;
    std::transform(params.outAxes.begin(), params.outAxes.end(), std::back_inserter(shape), [&params](int axis) {
      return params.ctx.allVectorSizeValues[axis] ? ShapedType::kDynamic : params.ctx.allVectorSizes[axis];
    });
    params.refVecType = npuvector::NPUVectorType::get(shape, params.refVecType.getElementType());
    return true;
  }

  for (unsigned j = 0, e = params.vecOperands.size(); j < e; ++j) {
    if (isScalarSlotOperand(j, params.scalarIndices)) {
      continue;
    }
    Value vo = params.vecOperands[j];
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(vo.getType());
    if (!nvt || static_cast<unsigned>(nvt.getRank()) != refRank) {
      continue;
    }
    auto it = params.ctx.valueDimOrder.find(vo);
    if (it != params.ctx.valueDimOrder.end() && it->second.size() == refRank) {
      params.outAxes.assign(it->second.begin(), it->second.end());
      return true;
    }
  }
  emitError(params.arithOp->getLoc()) << "npuvector-vectorize: rank-lift broadcast needs a peer !npuvector at rank "
                                      << refRank << " with non-empty valueDimOrder on this op's operands (no fallback)";
  return false;
}

struct BroadcastScalarSlotsParams {
  SmallVectorImpl<Value> &vecOperands;
  const SmallVectorImpl<unsigned> &scalarIndices;
  bool haveRefVec = false;
  npuvector::NPUVectorType refVecType;
  llvm::ArrayRef<int> broadcastCtxAxes;
  LoopVectorizationCtx &ctx;
};

static bool broadcastScalarSlotsToRef(const BroadcastScalarSlotsParams &params) {
  npuvector::NPUVectorType target = params.haveRefVec ? params.refVecType : npuvector::NPUVectorType{};
  for (unsigned idx : params.scalarIndices) {
    Value broadcasted = vectorizeBroadcastScalar(params.vecOperands[idx], params.ctx, target, params.broadcastCtxAxes);
    if (broadcasted == nullptr) {
      return false;
    }
    params.vecOperands[idx] = broadcasted;
  }
  return true;
}

static bool alignOperandRanksToRef(SmallVectorImpl<Value> &vecOperands, npuvector::NPUVectorType refVecType,
                                   llvm::ArrayRef<int> broadcastCtxAxes, LoopVectorizationCtx &ctx) {
  for (auto &vecOperand : vecOperands) {
    auto cur = mlir::dyn_cast<npuvector::NPUVectorType>(vecOperand.getType());
    if (!cur) {
      continue;
    }
    auto want = npuvector::NPUVectorType::get(refVecType.getShape(), cur.getElementType());
    if (want == cur) {
      continue;
    }
    Value aligned = vectorizeBroadcastScalar(vecOperand, ctx, refVecType, broadcastCtxAxes);
    if (!aligned) {
      return false;
    }
    vecOperand = aligned;
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
    if (!mlir::isa<npuvector::NPUVectorType>(vo.getType())) {
      continue;
    }
    auto ordIter = ctx.valueDimOrder.find(vo);
    if (ordIter == ctx.valueDimOrder.end()) {
      continue;
    }
    // Copy before operator[](result): inserting 'result' may rehash DenseMap and invalidate ordIter / references.
    SmallVector<int> orderCopy = ordIter->second;
    if (orderCopy.empty()) {
      continue;
    }
    if (expectRank != 0 && orderCopy.size() != expectRank) {
      continue;
    }
    ctx.valueDimOrder[result] = std::move(orderCopy);
    break;
  }
}

static void reportVectorizeArithOpFailure(Operation *op, llvm::StringRef reason, const LoopVectorizationCtx &ctx) {
  if (!ctx.vf1FuncLevelNoAnchor) {
    return;
  }
  emitError(op->getLoc()) << "npuvector-vectorize: vectorizeArithOp failed on '" << op->getName() << "': " << reason;
}

static Value vectorizeArithOp(Operation *op, LoopVectorizationCtx &ctx) {
  Location loc = op->getLoc();
  if (hasIndexResult(*op)) {
    reportVectorizeArithOpFailure(op, "index-typed result (use scalar clone path instead of vectorizeArithOp)", ctx);
    return nullptr;
  }
  mergeAncestorValueDimOrderMissingKeys(ctx);
  SmallVector<Value, kSmallVectorSizeFour> vecOperands;
  SmallVector<unsigned> scalarIndices;
  collectArithVecOperands(op, ctx, vecOperands, scalarIndices);

  npuvector::NPUVectorType refVecType;
  const bool haveRefVec = pickHighestRankNpuVecType(vecOperands, refVecType);
  SmallVector<int, kSmallVectorSizeEight> broadcastAxesBuf;
  llvm::ArrayRef<int> broadcastCtxAxes;
  if (haveRefVec) {
    if (!lookupPeerBroadcastAxes(
          LookupPeerBroadcastAxesParams{op, ctx, vecOperands, scalarIndices, refVecType, broadcastAxesBuf})) {
      return nullptr;
    }
    broadcastCtxAxes = broadcastAxesBuf;
  }

  if (!broadcastScalarSlotsToRef(
        BroadcastScalarSlotsParams{vecOperands, scalarIndices, haveRefVec, refVecType, broadcastCtxAxes, ctx})) {
    reportVectorizeArithOpFailure(op, "broadcastScalarSlotsToRef (vectorizeBroadcastScalar failed)", ctx);
    return nullptr;
  }
  if (haveRefVec && !alignOperandRanksToRef(vecOperands, refVecType, broadcastCtxAxes, ctx)) {
    reportVectorizeArithOpFailure(op, "alignOperandRanksToRef", ctx);
    return nullptr;
  }

  SmallVector<Type, kSmallVectorSizeFour> vecResultTypes;
  for (Value result : op->getResults()) {
    Type scalarType = result.getType();
    npuvector::NPUVectorType vecType =
      haveRefVec ? npuvector::NPUVectorType::get(refVecType.getShape(), scalarType) : ctx.getVectorType(scalarType);
    vecResultTypes.push_back(vecType);
  }

  Operation *vecOp = createRenamedOrSameNameVecArith(op, loc, vecOperands, vecResultTypes, ctx);
  if ((vecOp == nullptr) || vecOp->getNumResults() == 0) {
    reportVectorizeArithOpFailure(op,
                                  (vecOp == nullptr)
                                    ? "createRenamedOrSameNameVecArith returned null (unsupported op or "
                                      "invalid vector operands/result types for this named op)"
                                    : "createRenamedOrSameNameVecArith produced op with zero results",
                                  ctx);
    return nullptr;
  }

  Value vecResult = vecOp->getResult(0);
  propagateValueDimOrderToFirstOperandWithOrder(ctx, vecOperands, vecResult);
  return vecResult;
}

static int64_t findDestDimForGlobalCtxAxis(llvm::ArrayRef<int> resultDimToCtxAxis, int64_t dstRank, int globalCtxAxis) {
  if (static_cast<size_t>(dstRank) != resultDimToCtxAxis.size()) {
    return -1;
  }
  for (int64_t resultDim = 0; resultDim < dstRank; ++resultDim) {
    if (resultDimToCtxAxis[static_cast<unsigned>(resultDim)] == globalCtxAxis) {
      return resultDim;
    }
  }
  return -1;
}

static bool resolveBroadcastSrcToDestAxes(Value sourceVal, const LoopVectorizationCtx &ctx,
                                          llvm::ArrayRef<int> resultDimToCtxAxis, int64_t dstRank,
                                          SmallVectorImpl<int64_t> &outAxes) {
  outAxes.clear();
  auto srcNvt = mlir::dyn_cast<npuvector::NPUVectorType>(sourceVal.getType());
  if (!srcNvt) {
    return false;
  }
  const auto srcRank = static_cast<int64_t>(srcNvt.getRank());
  if (srcRank <= 0 || srcRank >= dstRank) {
    return false;
  }

  auto orderIter = ctx.valueDimOrder.find(sourceVal);
  if (orderIter == ctx.valueDimOrder.end()) {
    return false;
  }
  const SmallVector<int> &srcDimToGlobalAxis = orderIter->second;
  if (static_cast<int64_t>(srcDimToGlobalAxis.size()) != srcRank) {
    return false;
  }

  const auto ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  if (dstRank > static_cast<int64_t>(ctxRank)) {
    return false;
  }
  const unsigned ctxOff = ctxRank - static_cast<unsigned>(dstRank);

  llvm::SmallDenseSet<int64_t> usedDestDim;
  for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
    const int globalCtxAxis = srcDimToGlobalAxis[static_cast<unsigned>(srcDim)];
    if (globalCtxAxis < 0 || static_cast<unsigned>(globalCtxAxis) >= ctxRank) {
      return false;
    }

    int64_t destDim = -1;
    if (!resultDimToCtxAxis.empty()) {
      destDim = findDestDimForGlobalCtxAxis(resultDimToCtxAxis, dstRank, globalCtxAxis);
    } else {
      const auto axisUnsigned = static_cast<unsigned>(globalCtxAxis);
      if (axisUnsigned < ctxOff) {
        return false;
      }
      const unsigned relative = axisUnsigned - ctxOff;
      if (relative >= static_cast<unsigned>(dstRank)) {
        return false;
      }
      destDim = static_cast<int64_t>(relative);
    }
    if (destDim < 0) {
      return false;
    }
    if (!usedDestDim.insert(destDim).second) {
      return false;
    }
    outAxes.push_back(destDim);
  }
  return static_cast<int64_t>(outAxes.size()) == srcRank;
}

struct GatherBroadcastExtentParams {
  Location loc;
  LoopVectorizationCtx &ctx;
  unsigned outRank = 0;
  unsigned ctxRank = 0;
  llvm::ArrayRef<int> resultDimToCtxAxis;
  SmallVectorImpl<Value> &dynamicSizes;
  SmallVectorImpl<Value> &maxSizes;
};

static bool gatherBroadcastExtentOperands(const GatherBroadcastExtentParams &params) {
  const unsigned ctxOff = params.ctxRank - params.outRank;
  params.dynamicSizes.reserve(params.outRank);
  params.maxSizes.reserve(params.outRank);
  for (unsigned i = 0; i < params.outRank; ++i) {
    unsigned axisIdx =
      params.resultDimToCtxAxis.empty() ? (ctxOff + i) : static_cast<unsigned>(params.resultDimToCtxAxis[i]);
    if (!params.resultDimToCtxAxis.empty() && axisIdx >= params.ctxRank) {
      return false;
    }
    params.maxSizes.push_back(params.ctx.valueMapping.lookupOrDefault(params.ctx.allMaxStepValues[axisIdx]));
    params.dynamicSizes.push_back(getRuntimeVectorSizeValue(params.ctx, axisIdx, params.loc));
  }
  return true;
}

struct ResolveRankLiftBroadcastAxesParams {
  Location loc;
  Value scalarVal;
  LoopVectorizationCtx &ctx;
  llvm::ArrayRef<int> resultDimToCtxAxis;
  int64_t srcRank = 0;
  int64_t dstRank = 0;
  SmallVectorImpl<int64_t> &dimAxes;
};

static bool resolveRankLiftBroadcastAxes(const ResolveRankLiftBroadcastAxesParams &params) {
  mergeAncestorValueDimOrderMissingKeys(params.ctx);
  if (resolveBroadcastSrcToDestAxes(params.scalarVal, params.ctx, params.resultDimToCtxAxis, params.dstRank,
                                    params.dimAxes)) {
    return true;
  }
  if (Operation *defOp = params.scalarVal.getDefiningOp()) {
    emitError(defOp->getLoc()) << "npuvector-vectorize: cannot infer npuvector.broadcast `dimension` for rank lift ("
                               << params.srcRank << "D -> " << params.dstRank
                               << "D): resolveBroadcastSrcToDestAxes failed; align source and peer "
                                  "valueDimOrder / resultDimToCtxAxis (broadcastCtxAxes)";
  } else {
    emitError(params.loc) << "npuvector-vectorize: cannot infer npuvector.broadcast `dimension` for rank lift ("
                          << params.srcRank << "D -> " << params.dstRank
                          << "D): resolveBroadcastSrcToDestAxes failed; align source and peer "
                             "valueDimOrder / resultDimToCtxAxis (broadcastCtxAxes)";
  }
  return false;
}

static Value vectorizeBroadcastScalar(Value scalarVal, LoopVectorizationCtx &ctx, npuvector::NPUVectorType targetType,
                                      llvm::ArrayRef<int> resultDimToCtxAxis) {
  Type elemType;
  if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(scalarVal.getType())) {
    elemType = nvt.getElementType();
  } else {
    elemType = scalarVal.getType();
  }

  if (mlir::isa<npuvector::NPUVectorType>(scalarVal.getType())) {
    if (targetType) {
      auto want = npuvector::NPUVectorType::get(targetType.getShape(), elemType);
      if (want == scalarVal.getType()) {
        return scalarVal;
      }
    } else {
      return scalarVal;
    }
  }

  Location loc = scalarVal.getLoc();

  npuvector::NPUVectorType vecType =
    targetType ? npuvector::NPUVectorType::get(targetType.getShape(), elemType) : ctx.getVectorType(elemType);

  const auto outRank = static_cast<unsigned>(vecType.getRank());
  const auto ctxRank = static_cast<unsigned>(ctx.allVectorSizes.size());
  if (outRank > ctxRank) {
    return {};
  }
  if (!resultDimToCtxAxis.empty() && resultDimToCtxAxis.size() != outRank) {
    return {};
  }

  SmallVector<Value> dynamicSizes;
  SmallVector<Value> maxSizes;
  if (!gatherBroadcastExtentOperands(
        GatherBroadcastExtentParams{loc, ctx, outRank, ctxRank, resultDimToCtxAxis, dynamicSizes, maxSizes})) {
    return {};
  }

  const auto dstRank = static_cast<int64_t>(vecType.getRank());
  int64_t srcRank = 0;
  if (auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(scalarVal.getType())) {
    srcRank = static_cast<int64_t>(nvt.getRank());
  }
  SmallVector<int64_t> dimAxes;
  if (srcRank > 0 && srcRank < dstRank &&
      !resolveRankLiftBroadcastAxes(
        ResolveRankLiftBroadcastAxesParams{loc, scalarVal, ctx, resultDimToCtxAxis, srcRank, dstRank, dimAxes})) {
    return {};
  }

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
  for (const auto &kv : ctx.valueMapping.getValueMap()) {
    mapper.map(kv.first, kv.second);
  }

  Operation *clonedOp = ctx.builder.clone(op, mapper);

  clonedOp->removeAttr(kSkipVectorizeAttr);

  for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
    Value mappedValue = ctx.valueMapping.lookupOrDefault(operand);
    clonedOp->setOperand(idx, mappedValue);
  }

  for (unsigned i = 0, e = op.getNumResults(); i != e; ++i) {
    ctx.valueMapping.map(op.getResult(i), clonedOp->getResult(i));
  }
}

static LogicalResult vectorizeRegion(Region &region, LoopVectorizationCtx &ctx);

enum class SignedEuclideanDivKind { FloorTowardNegInfinity, CeilTowardPosInfinity };

static int64_t signedDivEuclidean(int64_t numerator, int64_t denominator, SignedEuclideanDivKind kind) {
  assert(denominator != 0 && "signedDivEuclidean: zero denominator");
  const int64_t quotient = numerator / denominator;
  const int64_t remainder = numerator % denominator;
  if (kind == SignedEuclideanDivKind::FloorTowardNegInfinity) {
    if (remainder != 0 && ((remainder > 0) != (denominator > 0))) {
      return quotient - 1;
    }
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
  if (!bounds.recognized || bounds.predicateEmptyOnIntegers) {
    return;
  }
  if (bounds.hasFiniteLowerInclusive && bounds.lowerInclusive < 0) {
    bounds.lowerInclusive = 0;
  }
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

struct DispatchNonzeroCompareParams {
  arith::CmpIPredicate effectivePredicate = arith::CmpIPredicate::eq;
  int64_t a = 0;
  int64_t K = 0;
  int64_t kMin = 0;
  int64_t kMax = 0;
  CompareSliceBounds &result;
};

static void dispatchNonzeroCompare(const DispatchNonzeroCompareParams &params) {
  switch (params.effectivePredicate) {
    case arith::CmpIPredicate::eq:
      fillEqSlice(params.a, params.K, params.result);
      return;
    case arith::CmpIPredicate::sge:
      fillSgeSlice(params.a, params.K, params.result);
      return;
    case arith::CmpIPredicate::sle:
      fillSleSlice(params.a, params.K, params.result);
      return;
    case arith::CmpIPredicate::sgt:
      fillSgtSlice(params.a, params.K, params.kMin, params.kMax, params.result);
      return;
    case arith::CmpIPredicate::slt:
      fillSltSlice(params.a, params.K, params.kMin, params.kMax, params.result);
      return;
    default:
      params.result.recognized = false;
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
      if (lhsConstant == R) {
        markCompareSliceFullLineUnconstrained(result);
      } else {
        markCompareSliceEmptyOnIntegers(result);
      }
      return result;
    case arith::CmpIPredicate::slt:
      if (lhsConstant < R) {
        markCompareSliceFullLineUnconstrained(result);
      } else {
        markCompareSliceEmptyOnIntegers(result);
      }
      return result;
    case arith::CmpIPredicate::sle:
      if (lhsConstant <= R) {
        markCompareSliceFullLineUnconstrained(result);
      } else {
        markCompareSliceEmptyOnIntegers(result);
      }
      return result;
    case arith::CmpIPredicate::sgt:
      if (lhsConstant > R) {
        markCompareSliceFullLineUnconstrained(result);
      } else {
        markCompareSliceEmptyOnIntegers(result);
      }
      return result;
    case arith::CmpIPredicate::sge:
      if (lhsConstant >= R) {
        markCompareSliceFullLineUnconstrained(result);
      } else {
        markCompareSliceEmptyOnIntegers(result);
      }
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

  if (originalPredicate == arith::CmpIPredicate::ne) {
    return result;
  }

  arith::CmpIPredicate effectivePredicate = originalPredicate;
  const int64_t a = linearCoefficientOfInductionVar;
  const int64_t b = affineConstantOffset;
  const int64_t R = rightHandSideIndexConstant;

  if (originalPredicate == arith::CmpIPredicate::ult || originalPredicate == arith::CmpIPredicate::ule ||
      originalPredicate == arith::CmpIPredicate::ugt || originalPredicate == arith::CmpIPredicate::uge) {
    if (a < 0 || b < 0 || R < 0) {
      return result;
    }
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
  if (a == -1 && (K == kMin || K == kMax)) {
    return result;
  }

  if (a == 0) {
    return fillZeroCoefficientCompare(effectivePredicate, b, R);
  }

  dispatchNonzeroCompare(DispatchNonzeroCompareParams{effectivePredicate, a, K, kMin, kMax, result});
  return result;
}

static bool decomposeAffineOneDimension(AffineExpr affineExpr, int64_t &coefficientOfDim0, int64_t &constantOffset);

struct AffineBinaryChildParts {
  int64_t &leftCoeff;
  int64_t &leftConst;
  int64_t &rightCoeff;
  int64_t &rightConst;
};

static bool decomposeAffineBinaryChildren(AffineExpr lhs, AffineExpr rhs, AffineBinaryChildParts parts) {
  return decomposeAffineOneDimension(lhs, parts.leftCoeff, parts.leftConst) &&
         decomposeAffineOneDimension(rhs, parts.rightCoeff, parts.rightConst);
}

static bool decomposeAffineOneDimension(AffineExpr affineExpr, int64_t &coefficientOfDim0, int64_t &constantOffset) {
  coefficientOfDim0 = 0;
  constantOffset = 0;
  if (auto constantExpr = dyn_cast<AffineConstantExpr>(affineExpr)) {
    constantOffset = constantExpr.getValue();
    return true;
  }
  if (auto dimExpr = dyn_cast<AffineDimExpr>(affineExpr)) {
    if (dimExpr.getPosition() != 0) {
      return false;
    }
    coefficientOfDim0 = 1;
    return true;
  }
  auto binaryExpr = dyn_cast<AffineBinaryOpExpr>(affineExpr);
  if (!binaryExpr) {
    return false;
  }
  if (binaryExpr.getKind() == AffineExprKind::Add) {
    int64_t leftCoeff = 0;
    int64_t leftConst = 0;
    int64_t rightCoeff = 0;
    int64_t rightConst = 0;
    if (!decomposeAffineBinaryChildren(binaryExpr.getLHS(), binaryExpr.getRHS(),
                                       AffineBinaryChildParts{leftCoeff, leftConst, rightCoeff, rightConst})) {
      return false;
    }
    coefficientOfDim0 = leftCoeff + rightCoeff;
    constantOffset = leftConst + rightConst;
    return true;
  }
  if (binaryExpr.getKind() == AffineExprKind::Mul) {
    int64_t leftCoeff = 0;
    int64_t leftConst = 0;
    int64_t rightCoeff = 0;
    int64_t rightConst = 0;
    if (!decomposeAffineBinaryChildren(binaryExpr.getLHS(), binaryExpr.getRHS(),
                                       AffineBinaryChildParts{leftCoeff, leftConst, rightCoeff, rightConst})) {
      return false;
    }
    if (leftCoeff != 0 && rightCoeff != 0) {
      return false;
    }
    coefficientOfDim0 = leftCoeff * rightConst + rightCoeff * leftConst;
    constantOffset = leftConst * rightConst;
    return true;
  }
  return false;
}

static bool definitionGraphContainsValue(Value conditionSsaValue, Value targetSsaValue) {
  SmallVector<Value, kSmallVectorSizeEight> worklist;
  DenseSet<Value> visited;
  worklist.push_back(conditionSsaValue);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == targetSsaValue) {
      return true;
    }
    if (!visited.insert(current).second) {
      continue;
    }
    Operation *definingOp = current.getDefiningOp();
    if (definingOp == nullptr) {
      continue;
    }
    worklist.insert(worklist.end(), definingOp->operand_begin(), definingOp->operand_end());
  }
  return false;
}

static LogicalResult vectorizeRegionBodyOnly(Region &region, LoopVectorizationCtx &ctx) {
  Block *block = &region.front();
  for (Operation &op : block->without_terminator()) {
    if (failed(vectorizeOneOp(op, ctx))) {
      return failure();
    }
  }
  return success();
}

static void emitVectorizedStore(memref::StoreOp storeOp, Value vecVal, LoopVectorizationCtx &ctx) {
  Location loc = storeOp.getLoc();
  SmallVector<int> storeDimOrder = collectStoreDimOrder(storeOp, ctx);
  if (!mlir::isa<npuvector::NPUVectorType>(vecVal.getType())) {
    storeOp.emitError("npuvector-vectorize: vectorized store value must be NPUVectorType");
    return;
  }
  if (!reorderStoreVectorForIndices(storeOp, ctx, loc, storeDimOrder, vecVal)) {
    return;
  }

  SmallVector<Value> indices;
  ValueRange rawIndices = storeOp.getIndices();
  indices.reserve(rawIndices.size());
  std::transform(rawIndices.begin(), rawIndices.end(), std::back_inserter(indices),
                 [&ctx](Value idx) { return ctx.valueMapping.lookupOrDefault(idx); });
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(storeOp.getMemRef());
  if (!createTransferWriteWithPermutationMap(
        CreateTransferWriteParams{storeOp, ctx, loc, vecVal, mappedMemRef, ValueRange(indices), storeDimOrder})) {
    return;
  }
}

static bool emitIfSlice(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc, Value inductionVar,
                        CompareSliceBounds sliceBounds) {
  OpBuilder &opBuilder = ctx.builder;
  int sliceAxisDimI = ctx.getVectorDimForIV(inductionVar);
  unsigned dimForTileOnSliceAxis;
  if (sliceAxisDimI >= 0) {
    dimForTileOnSliceAxis = static_cast<unsigned>(sliceAxisDimI);
  } else if (!ctx.allVectorSizeValues.empty()) {
    dimForTileOnSliceAxis = static_cast<unsigned>(ctx.allVectorSizeValues.size() - 1U);
  } else {
    return false;
  }
  if (dimForTileOnSliceAxis >= ctx.allVectorSizeValues.size() || dimForTileOnSliceAxis >= ctx.allVectorSizes.size()) {
    return false;
  }

  Value tileLowerBoundOnVectorAxis = ctx.valueMapping.lookupOrDefault(inductionVar);
  Value vectorExtentAlongAxisForThisTile;
  if (ctx.allVectorSizeValues[dimForTileOnSliceAxis]) {
    vectorExtentAlongAxisForThisTile = ctx.valueMapping.lookupOrDefault(ctx.allVectorSizeValues[dimForTileOnSliceAxis]);
  } else {
    vectorExtentAlongAxisForThisTile =
      opBuilder.create<arith::ConstantIndexOp>(loc, ctx.allVectorSizes[dimForTileOnSliceAxis]);
  }
  Value tileHalfOpenUpperBound =
    createAffineAdd(opBuilder, loc, tileLowerBoundOnVectorAxis, vectorExtentAlongAxisForThisTile);

  Value sliceLowerBoundOnVectorAxis = tileLowerBoundOnVectorAxis;
  if (sliceBounds.hasFiniteLowerInclusive) {
    sliceLowerBoundOnVectorAxis =
      createAffineMaxWithConstant(opBuilder, loc, tileLowerBoundOnVectorAxis, sliceBounds.lowerInclusive);
  }

  Value sliceHalfOpenUpperBound = tileHalfOpenUpperBound;
  if (sliceBounds.hasFiniteUpperExclusive) {
    sliceHalfOpenUpperBound =
      createAffineMinWithConstant(opBuilder, loc, tileHalfOpenUpperBound, sliceBounds.upperExclusive);
  }

  Value sliceHalfOpenUpperClampedNotBelowSliceLower =
    createAffineMax(opBuilder, loc, sliceHalfOpenUpperBound, sliceLowerBoundOnVectorAxis);
  Value vectorLengthForThenRegionAlongAxis =
    createAffineSub(opBuilder, loc, sliceHalfOpenUpperClampedNotBelowSliceLower, sliceLowerBoundOnVectorAxis);
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

struct EmitIfSliceWithElseParams {
  mutable scf::IfOp ifOp;
  LoopVectorizationCtx &ctx;
  Location loc;
  Value inductionVar;
  CompareSliceBounds sliceBounds;
  mutable memref::StoreOp consumerStore;
};

static bool emitIfSliceWithElse(const EmitIfSliceWithElseParams &params) {
  OpBuilder &b = params.ctx.builder;
  int axisDimI = params.ctx.getVectorDimForIV(params.inductionVar);
  unsigned dTile;
  if (axisDimI >= 0) {
    dTile = static_cast<unsigned>(axisDimI);
  } else if (!params.ctx.allVectorSizeValues.empty()) {
    dTile = static_cast<unsigned>(params.ctx.allVectorSizeValues.size() - 1U);
  } else {
    return false;
  }
  if (dTile >= params.ctx.allVectorSizeValues.size() || dTile >= params.ctx.allVectorSizes.size()) {
    return false;
  }

  Value tileLb = params.ctx.valueMapping.lookupOrDefault(params.inductionVar);
  Value tileVF;
  if (params.ctx.allVectorSizeValues[dTile]) {
    tileVF = params.ctx.valueMapping.lookupOrDefault(params.ctx.allVectorSizeValues[dTile]);
  } else {
    tileVF = b.create<arith::ConstantIndexOp>(params.loc, params.ctx.allVectorSizes[dTile]);
  }
  Value tileUb = createAffineAdd(b, params.loc, tileLb, tileVF);

  Value thenSliceLb = tileLb;
  if (params.sliceBounds.hasFiniteLowerInclusive) {
    thenSliceLb = createAffineMaxWithConstant(b, params.loc, tileLb, params.sliceBounds.lowerInclusive);
  }
  Value thenSliceUb = tileUb;
  if (params.sliceBounds.hasFiniteUpperExclusive) {
    thenSliceUb = createAffineMinWithConstant(b, params.loc, tileUb, params.sliceBounds.upperExclusive);
  }
  thenSliceUb = createAffineMax(b, params.loc, thenSliceUb, thenSliceLb);
  Value thenLen = createAffineSub(b, params.loc, thenSliceUb, thenSliceLb);
  if (params.sliceBounds.predicateEmptyOnIntegers) {
    thenLen = b.create<arith::ConstantIndexOp>(params.loc, 0);
    thenSliceLb = tileLb;
  }

  Value c0 = b.create<arith::ConstantIndexOp>(params.loc, 0);

  unsigned axisDim = dTile;

  // --- then slice ---
  Value thenGuard = b.create<arith::CmpIOp>(params.loc, arith::CmpIPredicate::ne, thenLen, c0);
  auto thenIfOp = b.create<scf::IfOp>(params.loc, TypeRange{}, thenGuard, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(thenIfOp.thenBlock());
    LoopVectorizationCtx thenCtx = params.ctx;
    thenCtx.valueMapping.map(params.inductionVar, thenSliceLb);
    thenCtx.allVectorSizeValues[axisDim] = thenLen;
    if (failed(vectorizeRegionBodyOnly(params.ifOp.getThenRegion(), thenCtx))) {
      thenIfOp.erase();
      return false;
    }
    auto thenYield = cast<scf::YieldOp>(params.ifOp.thenBlock()->getTerminator());
    Value yieldScalar = thenYield.getOperand(0);
    Value yieldVec = thenCtx.valueMapping.lookupOrNull(yieldScalar);
    if (!yieldVec) {
      yieldVec = vectorizeBroadcastScalar(yieldScalar, thenCtx);
      if (!yieldVec) {
        thenIfOp.erase();
        return false;
      }
    }
    emitVectorizedStore(params.consumerStore, yieldVec, thenCtx);
  }
  b.setInsertionPointAfter(thenIfOp);

  // --- else slice ---
  Value elseSliceLb = thenSliceUb;
  Value rawElseLen = createAffineSub(b, params.loc, tileUb, elseSliceLb);
  Value elseLen = createAffineMaxWithConstant(b, params.loc, rawElseLen, 0);
  Value elseGuard = b.create<arith::CmpIOp>(params.loc, arith::CmpIPredicate::ne, elseLen, c0);
  auto elseIfOp = b.create<scf::IfOp>(params.loc, TypeRange{}, elseGuard, false);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(elseIfOp.thenBlock());
    LoopVectorizationCtx elseCtx = params.ctx;
    elseCtx.valueMapping.map(params.inductionVar, elseSliceLb);
    elseCtx.allVectorSizeValues[axisDim] = elseLen;
    auto elseYield = cast<scf::YieldOp>(params.ifOp.elseBlock()->getTerminator());
    Value elseScalar = elseYield.getOperand(0);
    Value elseVec = vectorizeBroadcastScalar(elseScalar, elseCtx);
    if (!elseVec) {
      elseIfOp.erase();
      return false;
    }
    emitVectorizedStore(params.consumerStore, elseVec, elseCtx);
  }
  b.setInsertionPointAfter(elseIfOp);

  params.ctx.absorbedOps.insert(params.consumerStore.getOperation());
  return true;
}

static bool tryRecognizeIVDependentSlice(Value condition, Value inductionVar, CompareSliceBounds &sliceBounds) {
  auto compareIntegerOp = condition.getDefiningOp<arith::CmpIOp>();
  if (!compareIntegerOp) {
    return false;
  }
  auto rhsConst = tryConstantIndex(compareIntegerOp.getRhs());
  if (!rhsConst) {
    return false;
  }
  auto affineApplyOp = compareIntegerOp.getLhs().getDefiningOp<affine::AffineApplyOp>();
  if (!affineApplyOp) {
    return false;
  }
  AffineMap affineMap = affineApplyOp.getAffineMap();
  if (affineMap.getNumDims() != 1 || affineMap.getNumSymbols() != 0 || affineMap.getNumResults() != 1 ||
      affineApplyOp.getDimOperands().size() != 1 || affineApplyOp.getDimOperands()[0] != inductionVar) {
    return false;
  }
  int64_t coeff = 0;
  int64_t offset = 0;
  if (!decomposeAffineOneDimension(affineMap.getResult(0), coeff, offset)) {
    return false;
  }
  sliceBounds = deriveLinearCompareBounds(compareIntegerOp.getPredicate(), coeff, offset, *rhsConst);
  return sliceBounds.recognized;
}

static memref::StoreOp findUniqueStoreConsumer(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != kUnaryOpOperandCount) {
    return nullptr;
  }
  Value result = ifOp.getResult(0);
  memref::StoreOp consumer = nullptr;
  for (Operation *user : result.getUsers()) {
    auto store = dyn_cast<memref::StoreOp>(user);
    if (!store || consumer) {
      return nullptr;
    }
    consumer = store;
  }
  return consumer;
}

static Value resolveIfDependentIV(Value condition, LoopVectorizationCtx &ctx) {
  Value currentIV = ctx.scalarLoop.getInductionVar();
  if (hasVectorizationAttr(ctx.scalarLoop.getOperation()) && definitionGraphContainsValue(condition, currentIV)) {
    return currentIV;
  }
  for (auto &[loopOp, dimIgnored] : ctx.allLoopToVectorDim) {
    (void)dimIgnored;
    auto ancestorFor = dyn_cast<scf::ForOp>(loopOp);
    if (!ancestorFor || ancestorFor == ctx.scalarLoop) {
      continue;
    }
    if (definitionGraphContainsValue(condition, ancestorFor.getInductionVar())) {
      return ancestorFor.getInductionVar();
    }
  }
  return {};
}

static Value vectorizeIfElseWithVectorResults(scf::IfOp ifOp, LoopVectorizationCtx &ctx, Location loc,
                                              Value vecCondition) {
  SmallVector<Type> vecResultTypes;
  TypeRange resultTypes = ifOp.getResultTypes();
  vecResultTypes.reserve(resultTypes.size());
  std::transform(resultTypes.begin(), resultTypes.end(), std::back_inserter(vecResultTypes),
                 [&ctx](Type elemTy) { return ctx.getVectorType(elemTy); });
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

  auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYield = cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  for (auto [resultIdx, vecResult] : llvm::enumerate(vecIfOp.getResults())) {
    Value thenVec = ctx.valueMapping.lookupOrDefault(thenYield.getOperand(resultIdx));
    Value elseVec = ctx.valueMapping.lookupOrDefault(elseYield.getOperand(resultIdx));
    auto thenOrderIt = ctx.valueDimOrder.find(thenVec);
    auto elseOrderIt = ctx.valueDimOrder.find(elseVec);
    if (thenOrderIt != ctx.valueDimOrder.end() && elseOrderIt != ctx.valueDimOrder.end() &&
        thenOrderIt->second == elseOrderIt->second) {
      ctx.valueDimOrder[vecResult] = thenOrderIt->second;
    }
  }
  return vecIfOp.getResult(0);
}

enum class IvDependentIfRewrite { None, Failed, Finished };

struct RewriteIvDependentIfSlicesParams {
  mutable scf::IfOp ifOp;
  LoopVectorizationCtx &ctx;
  Location loc;
  Value dependentIV;
  Value condition;
  bool hasElse = false;
  bool hasResults = false;
};

static IvDependentIfRewrite rewriteIvDependentIfSlices(const RewriteIvDependentIfSlicesParams &params) {
  CompareSliceBounds sliceBounds{};
  if (!tryRecognizeIVDependentSlice(params.condition, params.dependentIV, sliceBounds)) {
    return IvDependentIfRewrite::None;
  }
  if (params.hasElse && params.hasResults) {
    memref::StoreOp consumer = findUniqueStoreConsumer(params.ifOp);
    if (consumer == nullptr) {
      return IvDependentIfRewrite::Failed;
    }
    if (!emitIfSliceWithElse(
          EmitIfSliceWithElseParams{params.ifOp, params.ctx, params.loc, params.dependentIV, sliceBounds, consumer})) {
      return IvDependentIfRewrite::Failed;
    }
    return IvDependentIfRewrite::Finished;
  }
  if (!params.hasElse && !params.hasResults) {
    if (!emitIfSlice(params.ifOp, params.ctx, params.loc, params.dependentIV, sliceBounds)) {
      return IvDependentIfRewrite::Failed;
    }
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
    IvDependentIfRewrite ivRewrite = rewriteIvDependentIfSlices(
      RewriteIvDependentIfSlicesParams{ifOp, ctx, loc, dependentIV, condition, hasElse, hasResults});
    if (ivRewrite == IvDependentIfRewrite::Failed) {
      return nullptr;
    }
    if (ivRewrite == IvDependentIfRewrite::Finished) {
      return {};
    }
  }

  // Non-IV-dependent + else + results: keep a vectorized `scf.if` result and let
  // the consumer op (for example, memref.store) handle its own vectorization.
  if (hasElse && hasResults && !dependentIV) {
    Value vecCondition = ctx.valueMapping.lookupOrNull(condition);
    if (!vecCondition) {
      vecCondition = condition;
    }
    return vectorizeIfElseWithVectorResults(ifOp, ctx, loc, vecCondition);
  }

  if (hasElse || hasResults) {
    return nullptr;
  }

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

  return {};
}

static LogicalResult handleConstantOp(arith::ConstantOp constOp, LoopVectorizationCtx &ctx) {
  cloneScalarOp(*constOp.getOperation(), ctx);
  return success();
}

static bool allOperandsNonNpuVector(Operation &op, LoopVectorizationCtx &ctx) {
  if (op.getNumOperands() == 0) {
    return false;
  }
  return llvm::all_of(op.getOperands(), [&ctx](Value opnd) {
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
    if (consumer && ctx.absorbedOps.contains(consumer.getOperation())) {
      return success();
    }
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
    if (parentCtx.valueDimOrder.find(kv.first) == parentCtx.valueDimOrder.end()) {
      parentCtx.valueDimOrder[kv.first] = kv.second;
    }
  }
}

static void mergeAncestorValueDimOrderMissingKeys(LoopVectorizationCtx &ctx) {
  for (LoopVectorizationCtx *ancestor = ctx.parent; ancestor != nullptr; ancestor = ancestor->parent) {
    for (const auto &kv : ancestor->valueDimOrder) {
      if (ctx.valueDimOrder.find(kv.first) == ctx.valueDimOrder.end()) {
        ctx.valueDimOrder[kv.first] = kv.second;
      }
    }
  }
}

static void registerChildResults(LoopVectorizationCtx &child, LoopVectorizationCtx &parentCtx) {
  if (child.mode == VectorizationMode::Elementwise) {
    for (const auto &kv : child.valueMapping.getValueMap()) {
      if (!parentCtx.valueMapping.lookupOrNull(kv.first)) {
        parentCtx.valueMapping.map(kv.first, kv.second);
      }
    }
    mergeChildValueDimOrderIntoParent(child, parentCtx);
    for (const auto &kv : child.allocBypass) {
      parentCtx.allocBypass[kv.first] = kv.second;
    }
    for (const auto &kv : child.scratchMeta) {
      parentCtx.scratchMeta[kv.first] = kv.second;
    }
    return;
  }

  if (child.mode == VectorizationMode::ReductionX) {
    mergeChildValueDimOrderIntoParent(child, parentCtx);
    for (const auto &kv : child.allocBypass) {
      parentCtx.allocBypass[kv.first] = kv.second;
    }
    for (const auto &kv : child.scratchMeta) {
      parentCtx.scratchMeta[kv.first] = kv.second;
    }
    return;
  }

  if (child.vecLoop) {
    for (auto [scalarResult, vecResult] : llvm::zip(child.scalarLoop.getResults(), child.vecLoop.getResults())) {
      parentCtx.valueMapping.map(scalarResult, vecResult);
      // finalizeReductionY / finalize may `replaceAllUsesWith(scalarResult, vecResult)` before
      // parent walks remaining ops; users then hold vecResult directly. Parent lookup must resolve
      // vecResult as well (IRMapping only had scalarResult -> vecResult as key).
      if (!parentCtx.valueMapping.lookupOrNull(vecResult)) {
        parentCtx.valueMapping.map(vecResult, vecResult);
      }
    }
    mergeChildValueDimOrderIntoParent(child, parentCtx);
  }
}

static bool hasTaggedDescendant(scf::ForOp loop) {
  bool found = false;
  loop->walk([&](scf::ForOp forOp) {
    if (forOp == loop) {
      return WalkResult::advance();
    }
    if (hasVectorizationAttr(forOp)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static LogicalResult vectorizeScalarCarrierLoop(scf::ForOp nestedForOp, LoopVectorizationCtx &ctx) {
  if (nestedForOp.getNumRegionIterArgs() != 0 || nestedForOp.getNumResults() != 0) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  Value mappedLB = ctx.valueMapping.lookupOrDefault(nestedForOp.getLowerBound());
  Value mappedUB = ctx.valueMapping.lookupOrDefault(nestedForOp.getUpperBound());
  Value mappedStep = ctx.valueMapping.lookupOrDefault(nestedForOp.getStep());

  auto carrierLoop = ctx.builder.create<scf::ForOp>(nestedForOp.getLoc(), mappedLB, mappedUB, mappedStep);
  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(carrierLoop.getBody());

    LoopVectorizationCtx carrierCtx = ctx;
    carrierCtx.valueMapping.map(nestedForOp.getInductionVar(), carrierLoop.getInductionVar());
    if (failed(vectorizeRegion(nestedForOp.getRegion(), carrierCtx))) {
      carrierLoop.erase();
      return failure();
    }
  }
  ctx.builder.setInsertionPointAfter(carrierLoop);
  return success();
}

static LogicalResult handleNestedForOp(scf::ForOp nestedForOp, LoopVectorizationCtx &ctx) {
  int64_t nestedMaxStep = -1;
  VectorizationMode nestedMode = getVectorizationMode(nestedForOp, nestedMaxStep);
  if (nestedMode == VectorizationMode::None) {
    if (hasTaggedDescendant(nestedForOp)) {
      return vectorizeScalarCarrierLoop(nestedForOp, ctx);
    }
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }
  if (nestedMaxStep <= 0) {
    cloneScalarOp(*nestedForOp.getOperation(), ctx);
    return success();
  }

  if (isMergedReductionLoop(ctx) && llvm::is_contained(ctx.mergedReductionLoops, nestedForOp.getOperation())) {
    updateNestedLoopOperands(nestedForOp, ctx);
    LoopVectorizationCtx childCtx = LoopVectorizationCtx::createMergedReductionChild(ctx, nestedForOp);
    processLoop(childCtx);
    registerChildResults(childCtx, ctx);
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

  LoopVectorizationCtx childCtx = LoopVectorizationCtx::createChild(
    CreateChildParams{ctx, nestedForOp, nestedMode, childVecSize, childVecSizeVal, childMaxStep});

  processLoop(childCtx);
  registerChildResults(childCtx, ctx);
  return success();
}

/// True when static alloc shape matches the current vector tile extents.
static bool allocShapeMatchesTile(ArrayRef<int64_t> shape, const LoopVectorizationCtx &ctx) {
  if (static_cast<int64_t>(shape.size()) != ctx.getRank()) {
    return false;
  }
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i] != ctx.allVectorSizes[i]) {
      return false;
    }
  }
  return true;
}

static LogicalResult vectorizeAllocTileBypass(memref::AllocOp allocOp, Operation &op, LoopVectorizationCtx &ctx) {
  auto memrefType = allocOp.getType();
  if (memrefType.hasStaticShape() && ctx.getRank() > 0 && allocShapeMatchesTile(memrefType.getShape(), ctx)) {
    ctx.allocBypass[allocOp.getResult()] = Value();
    cloneScalarOp(op, ctx);
    return success();
  }
  cloneScalarOp(op, ctx);
  return success();
}

// Virtual scratch forwarding.

namespace {
struct NormalizedScratchAccess {
  Value root;
  SmallVector<OpFoldResult> rootIndices;
};
}  // namespace

static std::optional<OpFoldResult> addScratchIndexOFR(OpFoldResult lhs, OpFoldResult rhs, MLIRContext *context) {
  std::optional<int64_t> cl = getConstantIntValue(lhs);
  std::optional<int64_t> cr = getConstantIntValue(rhs);
  if (cl && *cl == 0) {
    return rhs;
  }
  if (cr && *cr == 0) {
    return lhs;
  }
  if (cl && cr) {
    return OpFoldResult(IntegerAttr::get(IndexType::get(context), *cl + *cr));
  }
  return std::nullopt;
}

static OpFoldResult zeroIndexOFR(MLIRContext *context) {
  return OpFoldResult(IntegerAttr::get(IndexType::get(context), 0));
}

static LogicalResult normalizeSubviewAccess(memref::SubViewOp subview, SmallVector<OpFoldResult> &curIdx,
                                            MLIRContext *context) {
  for (OpFoldResult stride : subview.getMixedStrides()) {
    std::optional<int64_t> constantStride = getConstantIntValue(stride);
    if (!constantStride || *constantStride != 1) {
      return failure();
    }
  }

  SmallVector<OpFoldResult> mixedOffsets = subview.getMixedOffsets();
  llvm::SmallBitVector dropped = subview.getDroppedDims();
  const unsigned srcRank = static_cast<unsigned>(subview.getSourceType().getRank());
  SmallVector<OpFoldResult> srcIdx(srcRank);
  unsigned resPos = 0;
  for (unsigned d = 0; d < srcRank; ++d) {
    OpFoldResult offset = mixedOffsets[d];
    if (dropped.test(d)) {
      srcIdx[d] = offset;
      continue;
    }
    if (resPos >= curIdx.size()) {
      return failure();
    }
    std::optional<OpFoldResult> sum = addScratchIndexOFR(offset, curIdx[resPos], context);
    if (!sum) {
      return failure();
    }
    srcIdx[d] = *sum;
    ++resPos;
  }
  if (resPos != curIdx.size()) {
    return failure();
  }
  curIdx = std::move(srcIdx);
  return success();
}

static LogicalResult normalizeCollapseShapeAccess(memref::CollapseShapeOp collapse, SmallVector<OpFoldResult> &curIdx,
                                                  MLIRContext *context) {
  auto srcType = collapse.getSrcType();
  ArrayRef<int64_t> srcShape = srcType.getShape();
  SmallVector<ReassociationIndices> reassoc = collapse.getReassociationIndices();
  if (reassoc.size() != curIdx.size()) {
    return failure();
  }

  SmallVector<OpFoldResult> srcIdx(srcType.getRank());
  for (unsigned g = 0; g < reassoc.size(); ++g) {
    SmallVector<int64_t> nonUnit;
    for (int64_t sd : reassoc[g]) {
      if (sd < 0 || sd >= static_cast<int64_t>(srcShape.size())) {
        return failure();
      }
      if (srcShape[sd] == 1) {
        continue;
      }
      if (srcShape[sd] == ShapedType::kDynamic) {
        return failure();
      }
      nonUnit.push_back(sd);
    }
    if (nonUnit.empty()) {
      std::optional<int64_t> ri = getConstantIntValue(curIdx[g]);
      if (!ri || *ri != 0) {
        return failure();
      }
      for (int64_t sd : reassoc[g]) {
        srcIdx[sd] = zeroIndexOFR(context);
      }
    } else if (nonUnit.size() == 1) {
      for (int64_t sd : reassoc[g]) {
        srcIdx[sd] = (sd == nonUnit[0]) ? curIdx[g] : zeroIndexOFR(context);
      }
    } else {
      return failure();
    }
  }
  curIdx = std::move(srcIdx);
  return success();
}

static LogicalResult normalizeExpandShapeAccess(memref::ExpandShapeOp expand, SmallVector<OpFoldResult> &curIdx,
                                                MLIRContext *context) {
  auto resultType = expand.getResultType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  SmallVector<ReassociationIndices> reassoc = expand.getReassociationIndices();
  if (static_cast<int64_t>(curIdx.size()) != resultType.getRank()) {
    return failure();
  }

  SmallVector<OpFoldResult> srcIdx(reassoc.size());
  for (unsigned g = 0; g < reassoc.size(); ++g) {
    SmallVector<int64_t> nonUnit;
    for (int64_t rd : reassoc[g]) {
      if (rd < 0 || rd >= static_cast<int64_t>(resultShape.size())) {
        return failure();
      }
      if (resultShape[rd] == 1) {
        continue;
      }
      if (resultShape[rd] == ShapedType::kDynamic) {
        return failure();
      }
      nonUnit.push_back(rd);
    }
    if (nonUnit.empty()) {
      for (int64_t rd : reassoc[g]) {
        std::optional<int64_t> ri = getConstantIntValue(curIdx[rd]);
        if (!ri || *ri != 0) {
          return failure();
        }
      }
      srcIdx[g] = zeroIndexOFR(context);
    } else if (nonUnit.size() == 1) {
      for (int64_t rd : reassoc[g]) {
        if (rd == nonUnit[0]) {
          continue;
        }
        std::optional<int64_t> ri = getConstantIntValue(curIdx[rd]);
        if (!ri || *ri != 0) {
          return failure();
        }
      }
      srcIdx[g] = curIdx[nonUnit[0]];
    } else {
      return failure();
    }
  }
  curIdx = std::move(srcIdx);
  return success();
}

static FailureOr<NormalizedScratchAccess> normalizeMemrefAccess(Value memref, ValueRange indices,
                                                                LoopVectorizationCtx &ctx) {
  MLIRContext *context = ctx.builder.getContext();
  SmallVector<OpFoldResult> curIdx = getAsOpFoldResult(indices);
  Value cur = memref;

  for (int guard = 0; guard < kMaxMemrefAccessNormalizeGuards; ++guard) {
    Operation *def = cur.getDefiningOp();
    if (!def) {
      break;
    }

    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource();
      continue;
    }
    if (auto msc = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      cur = msc.getSource();
      continue;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      if (failed(normalizeSubviewAccess(subview, curIdx, context))) {
        return failure();
      }
      cur = subview.getSource();
      continue;
    }

    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      if (failed(normalizeCollapseShapeAccess(collapse, curIdx, context))) {
        return failure();
      }
      cur = collapse.getSrc();
      continue;
    }

    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      if (failed(normalizeExpandShapeAccess(expand, curIdx, context))) {
        return failure();
      }
      cur = expand.getSrc();
      continue;
    }

    break;
  }

  NormalizedScratchAccess out;
  out.root = cur;
  out.rootIndices = std::move(curIdx);
  return out;
}

static bool isLocalNonEscapingScratch(Value root) {
  Operation *def = root.getDefiningOp();
  if (!def || !isa<memref::AllocOp, memref::AllocaOp>(def)) {
    return false;
  }

  SmallVector<Value> worklist{root};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second) {
      continue;
    }
    for (Operation *user : v.getUsers()) {
      if (isa<memref::LoadOp, memref::StoreOp, memref::DeallocOp>(user)) {
        continue;
      }
      if (isa<memref::SubViewOp, memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp,
              memref::MemorySpaceCastOp>(user)) {
        worklist.append(user->getResults().begin(), user->getResults().end());
        continue;
      }
      return false;
    }
  }
  return true;
}

static FailureOr<int64_t> getConstantVectorIndexLowerBound(Value idxVal) {
  auto blockArg = dyn_cast<BlockArgument>(idxVal);
  if (!blockArg) {
    return failure();
  }
  auto forOp = dyn_cast_or_null<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp) {
    return failure();
  }
  auto c = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  if (!c) {
    return failure();
  }
  return c.value();
}

static FailureOr<ScratchTileMeta> buildScratchTileMeta(const NormalizedScratchAccess &norm, MemRefType rootType,
                                                       ArrayRef<int> vdo, const LoopVectorizationCtx &ctx) {
  ScratchTileMeta meta;
  meta.tileAxisOrder.assign(vdo.begin(), vdo.end());
  meta.allocRank = rootType.getRank();
  meta.allocDimOfTileDim.assign(vdo.size(), -1);
  meta.tileDimToLoop.assign(vdo.size(), nullptr);
  meta.storeLowerBoundOfTileDim.assign(vdo.size(), 0);

  for (const auto &[loop, dim] : ctx.allLoopToVectorDim) {
    for (unsigned tileDim = 0; tileDim < vdo.size(); ++tileDim) {
      if (static_cast<int>(dim) == vdo[tileDim]) {
        meta.tileDimToLoop[tileDim] = loop;
      }
    }
  }

  for (unsigned d = 0; d < norm.rootIndices.size(); ++d) {
    Value idxVal = llvm::dyn_cast_if_present<Value>(norm.rootIndices[d]);
    if (!idxVal) {
      continue;
    }
    int axis = getVectorDimForIndex(idxVal, ctx);
    if (axis < 0 || !llvm::is_contained(vdo, axis)) {
      continue;
    }
    FailureOr<int64_t> lb = getConstantVectorIndexLowerBound(idxVal);
    if (failed(lb)) {
      return failure();
    }
    meta.channelDims.push_back({static_cast<int>(d), axis, *lb});
    for (unsigned tileDim = 0; tileDim < vdo.size(); ++tileDim) {
      if (vdo[tileDim] != axis) {
        continue;
      }
      meta.allocDimOfTileDim[tileDim] = static_cast<int>(d);
      meta.storeLowerBoundOfTileDim[tileDim] = *lb;
    }
  }
  if (meta.channelDims.empty()) {
    meta.storeRootIndices = norm.rootIndices;
  }
  return meta;
}

/// Plan for extract_slice / keep from a recorded scratch tile.
struct ScratchSlicePlan {
  SmallVector<bool> dropDim;
  SmallVector<Value> offsets;
  SmallVector<Value> sizes;
  /// Per tile dim: current-ctx axis id for kept dims, -1 if dropped.
  SmallVector<int> resultAxisForTileDim;
  SmallVector<int64_t> cacheKey;
  bool cacheable = true;
};

struct GetTileDimExtentParams {
  Value tile;
  npuvector::NPUVectorType tileType;
  unsigned tileDim = 0;
  LoopVectorizationCtx &ctx;
  Location loc;
  int curAxisHint = -1;
};

static Value getTileDimExtent(const GetTileDimExtentParams &params) {
  SmallVector<Value> tileExtents;
  if (gatherVectorDynExtents(params.tile, tileExtents) && params.tileDim < tileExtents.size() &&
      tileExtents[params.tileDim]) {
    return tileExtents[params.tileDim];
  }
  int64_t staticSize = params.tileType.getShape()[params.tileDim];
  if (staticSize != ShapedType::kDynamic) {
    return params.ctx.builder.create<arith::ConstantIndexOp>(params.loc, staticSize);
  }
  if (params.curAxisHint >= 0 && static_cast<unsigned>(params.curAxisHint) < params.ctx.allVectorSizes.size()) {
    return getRuntimeVectorSizeValue(params.ctx, static_cast<unsigned>(params.curAxisHint), params.loc);
  }
  return Value();
}

struct ComputeScratchSliceProjectionParams {
  const ScratchTileMeta &meta;
  const NormalizedScratchAccess &norm;
  unsigned tileRank = 0;
  Value tile;
  npuvector::NPUVectorType tileType;
  LoopVectorizationCtx &ctx;
  Location loc;
  ScratchSlicePlan &plan;
};

static bool computeScratchSliceProjection(const ComputeScratchSliceProjectionParams &params) {
  params.plan.dropDim.assign(params.tileRank, false);
  params.plan.offsets.assign(params.tileRank, Value());
  params.plan.sizes.assign(params.tileRank, Value());
  params.plan.resultAxisForTileDim.assign(params.tileRank, -1);
  params.plan.cacheKey.clear();
  params.plan.cacheable = true;

  if (params.meta.allocDimOfTileDim.size() != params.tileRank || params.meta.tileDimToLoop.size() != params.tileRank ||
      params.meta.storeLowerBoundOfTileDim.size() != params.tileRank) {
    return false;
  }

  Value c0 = params.ctx.builder.create<arith::ConstantIndexOp>(params.loc, 0);
  Value c1 = params.ctx.builder.create<arith::ConstantIndexOp>(params.loc, 1);

  for (unsigned tileDim = 0; tileDim < params.tileRank; ++tileDim) {
    const int allocDim = params.meta.allocDimOfTileDim[tileDim];
    if (allocDim < 0) {
      Operation *loop = params.meta.tileDimToLoop[tileDim];
      if (loop == nullptr) {
        return false;
      }
      auto axisIt = params.ctx.allLoopToVectorDim.find(loop);
      if (axisIt == params.ctx.allLoopToVectorDim.end()) {
        return false;
      }
      const int curAxis = static_cast<int>(axisIt->second);
      Value extent = getTileDimExtent(
        GetTileDimExtentParams{params.tile, params.tileType, tileDim, params.ctx, params.loc, curAxis});
      if (!extent) {
        return false;
      }
      params.plan.dropDim[tileDim] = false;
      params.plan.offsets[tileDim] = c0;
      params.plan.sizes[tileDim] = extent;
      params.plan.resultAxisForTileDim[tileDim] = curAxis;
      params.plan.cacheKey.push_back(-2);
      params.plan.cacheKey.push_back(curAxis);
      continue;
    }

    if (allocDim >= static_cast<int>(params.norm.rootIndices.size())) {
      return false;
    }
    OpFoldResult rootIndex = params.norm.rootIndices[allocDim];
    const int64_t storeLb = params.meta.storeLowerBoundOfTileDim[tileDim];

    if (std::optional<int64_t> c = getConstantIntValue(rootIndex)) {
      params.plan.dropDim[tileDim] = true;
      params.plan.offsets[tileDim] = params.ctx.builder.create<arith::ConstantIndexOp>(params.loc, *c - storeLb);
      params.plan.sizes[tileDim] = c1;
      params.plan.resultAxisForTileDim[tileDim] = -1;
      params.plan.cacheKey.push_back(*c - storeLb);
      continue;
    }

    Value idxVal = llvm::dyn_cast_if_present<Value>(rootIndex);
    if (!idxVal) {
      return false;
    }
    const int loadAxis = getVectorDimForIndex(idxVal, params.ctx);
    if (loadAxis < 0) {
      return false;
    }
    Value mappedIdx = params.ctx.valueMapping.lookupOrDefault(idxVal);
    Value offset = mappedIdx;
    if (storeLb != 0) {
      Value lbVal = params.ctx.builder.create<arith::ConstantIndexOp>(params.loc, storeLb);
      offset = createAffineSub(params.ctx.builder, params.loc, mappedIdx, lbVal);
    }
    Value tileSize = getRuntimeVectorSizeValue(params.ctx, static_cast<unsigned>(loadAxis), params.loc);
    params.plan.dropDim[tileDim] = false;
    params.plan.offsets[tileDim] = offset;
    params.plan.sizes[tileDim] = tileSize;
    params.plan.resultAxisForTileDim[tileDim] = loadAxis;
    params.plan.cacheKey.push_back(-1);
    params.plan.cacheKey.push_back(loadAxis);
    if (!tryConstantIndex(offset).has_value() || !tryConstantIndex(tileSize).has_value()) {
      params.plan.cacheable = false;
    }
  }
  return true;
}

struct AppendKeptScratchSliceDimParams {
  unsigned tileDim = 0;
  const ScratchSlicePlan &plan;
  npuvector::NPUVectorType tileType;
  SmallVectorImpl<int64_t> &keepDims;
  SmallVectorImpl<int> &resultDimOrder;
  SmallVectorImpl<int64_t> &sliceShape;
  bool &needsSlice;
};

/// Append one kept tile dim into slice plan outputs; set needsSlice when extent/offset is partial.
static void appendKeptScratchSliceDim(const AppendKeptScratchSliceDimParams &params) {
  params.keepDims.push_back(static_cast<int64_t>(params.tileDim));
  params.resultDimOrder.push_back(params.plan.resultAxisForTileDim[params.tileDim]);
  if (std::optional<int64_t> sizeConst = tryConstantIndex(params.plan.sizes[params.tileDim])) {
    params.sliceShape.push_back(*sizeConst);
    int64_t fullStatic = params.tileType.getShape()[params.tileDim];
    if (fullStatic != ShapedType::kDynamic && *sizeConst != fullStatic) {
      params.needsSlice = true;
    }
  } else {
    params.sliceShape.push_back(ShapedType::kDynamic);
    params.needsSlice = true;
  }
  if (std::optional<int64_t> offConst = tryConstantIndex(params.plan.offsets[params.tileDim])) {
    if (*offConst != 0) {
      params.needsSlice = true;
    }
  } else {
    params.needsSlice = true;
  }
}

static Value createForwardedScratchSlice(memref::LoadOp loadOp, Value tile, npuvector::NPUVectorType tileType,
                                         const ScratchSlicePlan &plan, LoopVectorizationCtx &ctx) {
  if (plan.cacheable) {
    for (auto &[cachedTile, cachedKey, cachedSlice] : ctx.scratchSliceCache) {
      if (cachedTile == tile && cachedKey == plan.cacheKey) {
        return cachedSlice;
      }
    }
  }

  Location loc = loadOp.getLoc();
  const unsigned tileRank = static_cast<unsigned>(tileType.getRank());
  if (plan.dropDim.size() != tileRank || plan.offsets.size() != tileRank || plan.sizes.size() != tileRank ||
      plan.resultAxisForTileDim.size() != tileRank) {
    return Value();
  }

  Value c1 = ctx.builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> strides(tileRank, c1);
  SmallVector<int64_t> keepDims;
  SmallVector<int> resultDimOrder;
  SmallVector<int64_t> sliceShape;
  bool anyDrop = false;
  bool needsSlice = false;
  for (unsigned t = 0; t < tileRank; ++t) {
    if (plan.dropDim[t]) {
      anyDrop = true;
      needsSlice = true;
      continue;
    }
    appendKeptScratchSliceDim(
      AppendKeptScratchSliceDimParams{t, plan, tileType, keepDims, resultDimOrder, sliceShape, needsSlice});
  }
  if (keepDims.empty()) {
    return Value();
  }

  if (!needsSlice && !anyDrop && keepDims.size() == tileRank) {
    ctx.valueDimOrder[tile] = resultDimOrder;
    return tile;
  }

  auto sliceType = npuvector::NPUVectorType::get(sliceShape, tileType.getElementType());
  Value slice = ctx.builder.create<npuvector::ExtractSliceOp>(loc, sliceType, tile, ValueRange(plan.offsets),
                                                              ValueRange(plan.sizes), ValueRange(strides),
                                                              ctx.builder.getDenseI64ArrayAttr(keepDims));
  ctx.valueDimOrder[slice] = resultDimOrder;
  if (plan.cacheable) {
    ctx.scratchSliceCache.push_back({tile, plan.cacheKey, slice});
  }
  return slice;
}

static bool scratchIndicesMatch(ArrayRef<OpFoldResult> storeIndices, ArrayRef<OpFoldResult> loadIndices) {
  if (storeIndices.size() != loadIndices.size()) {
    return false;
  }
  for (auto [storeIdx, loadIdx] : llvm::zip(storeIndices, loadIndices)) {
    std::optional<int64_t> storeConst = getConstantIntValue(storeIdx);
    std::optional<int64_t> loadConst = getConstantIntValue(loadIdx);
    if (storeConst || loadConst) {
      if (!storeConst || !loadConst || *storeConst != *loadConst) {
        return false;
      }
      continue;
    }
    Value storeValue = llvm::dyn_cast_if_present<Value>(storeIdx);
    Value loadValue = llvm::dyn_cast_if_present<Value>(loadIdx);
    if (!storeValue || storeValue != loadValue) {
      return false;
    }
  }
  return true;
}

static bool tryForwardVirtualScratchLoad(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
  if (ctx.scratchMeta.empty()) {
    return false;
  }

  FailureOr<NormalizedScratchAccess> norm = normalizeMemrefAccess(loadOp.getMemRef(), loadOp.getIndices(), ctx);
  if (failed(norm)) {
    return false;
  }
  Value root = norm->root;
  auto metaIt = ctx.scratchMeta.find(root);
  if (metaIt == ctx.scratchMeta.end()) {
    return false;
  }
  auto bypassIt = ctx.allocBypass.find(root);
  if (bypassIt == ctx.allocBypass.end() || !bypassIt->second) {
    return false;
  }

  Value tile = bypassIt->second;
  auto tileType = dyn_cast<npuvector::NPUVectorType>(tile.getType());
  if (!tileType) {
    return false;
  }
  const ScratchTileMeta &meta = metaIt->second;
  if (static_cast<int64_t>(norm->rootIndices.size()) != meta.allocRank) {
    return false;
  }
  if (static_cast<int64_t>(meta.tileAxisOrder.size()) != tileType.getRank()) {
    return false;
  }

  if (meta.channelDims.empty()) {
    if (!scratchIndicesMatch(meta.storeRootIndices, norm->rootIndices)) {
      return false;
    }
    ctx.valueMapping.map(loadOp.getResult(), tile);
    return true;
  }

  const unsigned tileRank = static_cast<unsigned>(tileType.getRank());
  ScratchSlicePlan plan;
  if (!computeScratchSliceProjection(
        ComputeScratchSliceProjectionParams{meta, *norm, tileRank, tile, tileType, ctx, loadOp.getLoc(), plan})) {
    return false;
  }
  Value slice = createForwardedScratchSlice(loadOp, tile, tileType, plan, ctx);
  if (!slice) {
    return false;
  }
  ctx.valueMapping.map(loadOp.getResult(), slice);
  return true;
}

static FailureOr<bool> tryRecordVirtualScratchStore(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  if (ctx.vf1FuncLevelNoAnchor) {
    return false;
  }

  Value storeValue = storeOp.getValue();
  Value vectorValue = ctx.valueMapping.lookupOrNull(storeValue);
  if (!vectorValue) {
    return false;
  }
  auto nvt = dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!nvt) {
    return false;
  }
  auto vdoIt = ctx.valueDimOrder.find(vectorValue);
  if (vdoIt == ctx.valueDimOrder.end() || vdoIt->second.empty()) {
    return false;
  }
  SmallVector<int> vdo = vdoIt->second;
  if (static_cast<int64_t>(vdo.size()) != nvt.getRank()) {
    return false;
  }

  SmallVector<int> storeAxes;
  for (Value idx : storeOp.getIndices()) {
    int d = getVectorDimForIndex(idx, ctx);
    if (d >= 0) {
      storeAxes.push_back(d);
    }
  }
  const bool hasMissingAxis = llvm::any_of(vdo, [&](int a) { return !llvm::is_contained(storeAxes, a); });
  if (!hasMissingAxis) {
    return false;
  }

  FailureOr<NormalizedScratchAccess> norm = normalizeMemrefAccess(storeOp.getMemRef(), storeOp.getIndices(), ctx);
  if (failed(norm)) {
    return false;
  }
  Value root = norm->root;
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType || static_cast<int64_t>(norm->rootIndices.size()) != rootType.getRank()) {
    return false;
  }
  if (!isLocalNonEscapingScratch(root)) {
    return false;
  }

  FailureOr<ScratchTileMeta> meta = buildScratchTileMeta(*norm, rootType, vdo, ctx);
  if (failed(meta)) {
    storeOp.emitError(
      "npuvector-vectorize: virtual scratch forwarding requires scratch indices "
      "mapped to vector axes to be direct scf.for induction variables with constant lower bounds");
    return failure();
  }
  ctx.allocBypass[root] = vectorValue;
  ctx.scratchMeta[root] = std::move(*meta);
  return true;
}

static LogicalResult vectorizeMemrefLoadLike(memref::LoadOp loadOp, LoopVectorizationCtx &ctx) {
  if (tryForwardVirtualScratchLoad(loadOp, ctx)) {
    return success();
  }

  if (!ctx.scratchMeta.empty()) {
    FailureOr<NormalizedScratchAccess> norm = normalizeMemrefAccess(loadOp.getMemRef(), loadOp.getIndices(), ctx);
    if (succeeded(norm) && ctx.scratchMeta.contains(norm->root)) {
      loadOp.emitError("npuvector-vectorize: virtual scratch load does not match recorded store");
      return failure();
    }
  }

  Value memRef = loadOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);
  auto bypassIt = ctx.allocBypass.find(mappedMemRef);
  if (bypassIt == ctx.allocBypass.end()) {
    bypassIt = ctx.allocBypass.find(memRef);
  }
  if (bypassIt != ctx.allocBypass.end() && bypassIt->second) {
    ctx.valueMapping.map(loadOp.getResult(), bypassIt->second);
    return success();
  }

  Value vecValue = vectorizeLoad(loadOp, ctx);
  if (!vecValue) {
    return failure();
  }
  ctx.valueMapping.map(loadOp.getResult(), vecValue);
  return success();
}

static LogicalResult vectorizeMemrefStoreLike(memref::StoreOp storeOp, LoopVectorizationCtx &ctx) {
  Value memRef = storeOp.getMemRef();
  Value mappedMemRef = ctx.valueMapping.lookupOrDefault(memRef);
  auto bypassIt = ctx.allocBypass.find(mappedMemRef);
  if (bypassIt == ctx.allocBypass.end()) {
    bypassIt = ctx.allocBypass.find(memRef);
  }
  if (bypassIt != ctx.allocBypass.end()) {
    Value storeValue = storeOp.getValue();
    Value vecVal = ctx.valueMapping.lookupOrNull(storeValue);
    if (vecVal) {
      bypassIt->second = vecVal;
    }
    return success();
  }

  FailureOr<bool> recordedVirtualScratch = tryRecordVirtualScratchStore(storeOp, ctx);
  if (failed(recordedVirtualScratch)) {
    return failure();
  }
  if (*recordedVirtualScratch) {
    return success();
  }

  if (!ctx.vf1FuncLevelNoAnchor) {
    bool hasVectorDim =
      llvm::any_of(storeOp.getIndices(), [&ctx](Value idx) { return getVectorDimForIndex(idx, ctx) >= 0; });
    Value mappedStoreValue = ctx.valueMapping.lookupOrNull(storeOp.getValue());
    bool hasMappedVectorValue = mappedStoreValue && mlir::isa<npuvector::NPUVectorType>(mappedStoreValue.getType());
    if (!hasVectorDim && !hasMappedVectorValue) {
      cloneScalarOp(*storeOp.getOperation(), ctx);
      return success();
    }
  }

  vectorizeStore(storeOp, ctx);
  return success();
}

static LogicalResult vectorizeOneOp(Operation &op, LoopVectorizationCtx &ctx) {
  if (ctx.absorbedOps.contains(&op)) {
    return success();
  }

  if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
    return vectorizeAllocTileBypass(allocOp, op, ctx);
  }

  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
    return vectorizeMemrefLoadLike(loadOp, ctx);
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
    return vectorizeMemrefStoreLike(storeOp, ctx);
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
    if (currentBlock == nullptr) {
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

static void recordReductionXCurrentTileSize(LoopVectorizationCtx &ctx) {
  if (ctx.mode != VectorizationMode::ReductionX || !ctx.vecLoop) {
    return;
  }

  auto dimIter = ctx.allLoopToVectorDim.find(ctx.scalarLoop.getOperation());
  if (dimIter == ctx.allLoopToVectorDim.end()) {
    return;
  }

  const unsigned axisIdx = dimIter->second;
  if (axisIdx >= ctx.allVectorSizes.size()) {
    return;
  }

  Location loc = ctx.vecLoop.getLoc();
  std::optional<int64_t> ubOpt = tryConstantIndex(ctx.vecLoop.getUpperBound());
  std::optional<int64_t> lbOpt = tryConstantIndex(ctx.vecLoop.getLowerBound());
  if (ubOpt && lbOpt && ((*ubOpt - *lbOpt) % ctx.allVectorSizes[axisIdx] == 0)) {
    return;
  }

  Value remaining = createAffineSub(ctx.builder, loc, ctx.vecLoop.getUpperBound(), ctx.vecLoop.getInductionVar());
  Value tileSize = createAffineMin(ctx.builder, loc, remaining, ctx.vecLoop.getStep());
  if (axisIdx >= ctx.currentVectorSizeValues.size()) {
    ctx.currentVectorSizeValues.resize(axisIdx + 1);
  }
  ctx.currentVectorSizeValues[axisIdx] = tileSize;
}

/// Map ReductionX/Y region iter_args to vecLoop args (with optional current-tile slice).
static void mapReductionRegionIterArgs(LoopVectorizationCtx &ctx) {
  std::optional<unsigned> currentTileAxis = getCurrentReductionTileAxis(ctx);
  for (unsigned idx = 0, e = ctx.scalarLoop.getNumRegionIterArgs(); idx < e; ++idx) {
    Value scalarArg = ctx.scalarLoop.getRegionIterArgs()[idx];
    Value vecArg = ctx.vecLoop.getRegionIterArgs()[idx];
    Value mappedArg = vecArg;

    Value vecInit = ctx.vecLoop.getInitArgs()[idx];
    auto dimOrderIter = ctx.valueDimOrder.find(vecInit);
    if (dimOrderIter != ctx.valueDimOrder.end()) {
      ctx.valueDimOrder[vecArg] = dimOrderIter->second;
      if (currentTileAxis) {
        CurrentTileSliceInfo sliceInfo;
        mappedArg = createCurrentTileSlice(vecArg, ctx, dimOrderIter->second, *currentTileAxis, &sliceInfo);
        if (mappedArg != vecArg) {
          ctx.iterArgTileSlices[idx] = std::move(sliceInfo);
        }
      }
    }
    ctx.valueMapping.map(scalarArg, mappedArg);
  }
}

static LogicalResult vectorizeLoopBody(LoopVectorizationCtx &ctx) {
  if (ctx.vecLoop) {
    ctx.builder.setInsertionPointToStart(ctx.vecLoop.getBody());
    ctx.valueMapping.map(ctx.scalarLoop.getInductionVar(), ctx.vecLoop.getInductionVar());
    recordReductionXCurrentTileSize(ctx);
  }

  if (ctx.allLoopToVectorDim.count(ctx.scalarLoop) == 0u) {
    const bool innerIvIsScalarForNestedVecAxis =
      (ctx.mode == VectorizationMode::ReductionY || ctx.mode == VectorizationMode::Broadcast) &&
      ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
    if (!innerIvIsScalarForNestedVecAxis) {
      ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;
    }
  }

  if (ctx.mode == VectorizationMode::ReductionX || ctx.mode == VectorizationMode::ReductionY) {
    mapReductionRegionIterArgs(ctx);
  }

  auto bodyOps = ctx.scalarLoop.getBody()->without_terminator();
  SmallVector<Operation *> opsToVectorize = llvm::map_to_vector(bodyOps, [](Operation &op) { return &op; });

  SmallVector<scf::ForOp> nestedLoopsToErase;

  for (Operation *op : opsToVectorize) {
    if ((op == nullptr) || op->getBlock() == nullptr) {
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
  if (!ctx.vecLoop || !ctx.scalarLoop.getInitArgs().empty()) {
    return;
  }
  if (ctx.mode != VectorizationMode::ReductionX && ctx.mode != VectorizationMode::ReductionY) {
    return;
  }
  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    return;
  }
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
  while (parent != nullptr) {
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

/// True if any ancestor `scf.for` (walking from `loop`'s parent) carries a vectorization attr.
static bool parentHasTaggedVectorLoop(scf::ForOp loop) {
  for (Operation *parent = loop->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
      if (hasVectorizationAttr(parentFor)) {
        return true;
      }
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
    if (!outNvt) {
      continue;
    }
    auto ordIter = ctx.valueDimOrder.find(yieldVal);
    if (ordIter == ctx.valueDimOrder.end() || ordIter->second.empty()) {
      continue;
    }
    const auto wantRank = static_cast<unsigned>(outNvt.getRank());
    if (ordIter->second.size() != wantRank) {
      continue;
    }
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

static npuvector::NPUVectorType buildReductionType(npuvector::NPUVectorType srcType, ArrayRef<int64_t> reductionDims) {
  llvm::SmallDenseSet<int64_t> reductionDimSet;
  for (int64_t dim : reductionDims) {
    reductionDimSet.insert(dim);
  }
  auto srcShape = srcType.getShape();
  SmallVector<int64_t> resultShape;
  for (unsigned i = 0; i < srcShape.size(); ++i) {
    if (!reductionDimSet.contains(static_cast<int64_t>(i))) {
      resultShape.push_back(srcShape[i]);
    }
  }
  return npuvector::NPUVectorType::get(resultShape, srcType.getElementType());
}

static LogicalResult inlineVectorize(LoopVectorizationCtx &ctx) {
  if (ctx.parent == nullptr) {
    ctx.builder.setInsertionPoint(ctx.scalarLoop);
  }

  if (ctx.allLoopToVectorDim.count(ctx.scalarLoop) == 0u) {
    ctx.allLoopToVectorDim[ctx.scalarLoop] = ctx.localDim;
  }

  Value iv = ctx.scalarLoop.getInductionVar();
  Value lb = (ctx.parent != nullptr) ? ctx.valueMapping.lookupOrDefault(ctx.scalarLoop.getLowerBound())
                                     : ctx.scalarLoop.getLowerBound();
  ctx.valueMapping.map(iv, lb);

  Block *body = ctx.scalarLoop.getBody();
  SmallVector<Operation *> opsToVec;
  auto bodyOps = body->without_terminator();
  opsToVec.reserve(static_cast<size_t>(std::distance(bodyOps.begin(), bodyOps.end())));
  std::transform(bodyOps.begin(), bodyOps.end(), std::back_inserter(opsToVec),
                 [](Operation &singleOp) { return &singleOp; });

  SmallVector<scf::ForOp> nestedLoopsToErase;
  for (Operation *op : opsToVec) {
    if ((op == nullptr) || op->getBlock() == nullptr) {
      continue;
    }
    if (auto nestedFor = dyn_cast<scf::ForOp>(op)) {
      int64_t nm = -1;
      if (getVectorizationMode(nestedFor, nm) != VectorizationMode::None) {
        nestedLoopsToErase.push_back(nestedFor);
      }
    }
    if (failed(vectorizeOneOp(*op, ctx))) {
      return failure();
    }
  }

  // Probe: nestedLoop.erase disabled (see vectorizeLoopBody); re-enable after bisect.
  (void)nestedLoopsToErase;
  if (ctx.parent == nullptr) {
    ctx.scalarLoop.erase();
  }
  return success();
}

struct MergeReductionXOriginalInitParams {
  Location loc;
  LoopVectorizationCtx &ctx;
  bool isMultiDim = false;
  unsigned totalRank = 0;
  Value &finalResult;
  unsigned laneIdx = 0;
};

static void mergeReductionXOriginalInit(const MergeReductionXOriginalInitParams &params) {
  if (params.laneIdx >= params.ctx.reductionKinds.size() || params.laneIdx >= params.ctx.origInits.size()) {
    return;
  }
  arith::AtomicRMWKind laneKind = params.ctx.reductionKinds[params.laneIdx];
  Value laneOrig = params.ctx.origInits[params.laneIdx];
  if (isNeutralElement(laneKind, laneOrig, params.ctx.builder)) {
    return;
  }
  if (params.isMultiDim) {
    unsigned ancestorRank = params.totalRank - 1;
    if (ancestorRank > 0) {
      SmallVector<int64_t> ancVecSizes(params.ctx.allVectorSizes.begin(),
                                       params.ctx.allVectorSizes.begin() + ancestorRank);
      SmallVector<Value> ancVecSizeValues(params.ctx.allVectorSizeValues.begin(),
                                          params.ctx.allVectorSizeValues.begin() + ancestorRank);
      SmallVector<Value> ancMaxSteps(params.ctx.allMaxStepValues.begin(),
                                     params.ctx.allMaxStepValues.begin() + ancestorRank);
      LoopVectorizationCtx ancCtx(LoopVectorizationInit{params.ctx.builder, std::move(ancVecSizes),
                                                        std::move(ancVecSizeValues), std::move(ancMaxSteps),
                                                        VectorizationMode::Elementwise, params.ctx.scalarLoop});
      ancCtx.parent = &params.ctx;
      for (const auto &kv : params.ctx.valueMapping.getValueMap()) {
        ancCtx.valueMapping.map(kv.first, kv.second);
      }
      for (const auto &kv : params.ctx.valueDimOrder) {
        ancCtx.valueDimOrder[kv.first] = kv.second;
      }
      Value initBroadcast = vectorizeBroadcastScalar(laneOrig, ancCtx);
      if (initBroadcast) {
        params.finalResult =
          combineReductionResults(params.ctx.builder, params.loc, params.finalResult, initBroadcast, laneKind);
      }
    } else {
      params.finalResult =
        combineReductionResults(params.ctx.builder, params.loc, params.finalResult, laneOrig, laneKind);
    }
    return;
  }
  auto resultType = mlir::dyn_cast<npuvector::NPUVectorType>(params.finalResult.getType());
  Value initBroadcast = resultType ? vectorizeBroadcastScalar(laneOrig, params.ctx, resultType) : Value();
  if (initBroadcast) {
    params.finalResult =
      combineReductionResults(params.ctx.builder, params.loc, params.finalResult, initBroadcast, laneKind);
  }
}

static Value buildReductionXLaneFinalResult(LoopVectorizationCtx &ctx, Location loc, unsigned laneIdx) {
  arith::AtomicRMWKind laneKind = ctx.reductionKinds[laneIdx];
  vector::CombiningKind combKind = convertToCombiningKind(laneKind);
  Value vectorAcc = ctx.vecLoop.getResult(laneIdx);
  auto fullVecType = mlir::cast<npuvector::NPUVectorType>(vectorAcc.getType());
  const unsigned totalRank = static_cast<unsigned>(fullVecType.getRank());

  SmallVector<int64_t> reductionDims;
  if (isMergedReductionRoot(ctx)) {
    reductionDims = buildAllReductionDims(totalRank);
  } else {
    reductionDims.push_back(static_cast<int64_t>(totalRank - 1));
  }

  auto reducedVecType = buildReductionType(fullVecType, reductionDims);
  auto reductionOp = ctx.builder.create<npuvector::ReductionOp>(loc, reducedVecType, combKind, vectorAcc, Value(),
                                                                ctx.builder.getDenseI64ArrayAttr(reductionDims),
                                                                arith::FastMathFlags::none);
  Value finalResult = reductionOp.getDest();

  const bool isMultiDim = totalRank > 1;
  mergeReductionXOriginalInit(MergeReductionXOriginalInitParams{loc, ctx, isMultiDim, totalRank, finalResult, laneIdx});
  return finalResult;
}

/// Build parent dim order for a reduction result (excludes this ctx's scalar loop).
/// Input: ctx with non-null parent. Output: sorted parent vector dims, or empty if incomplete.
static SmallVector<int> buildParentDimOrderForReductionResult(LoopVectorizationCtx &ctx) {
  SmallVector<int> parentDimOrder;
  Operation *scalarLoopOp = ctx.scalarLoop.getOperation();
  for (auto &[loop, localDim] : ctx.allLoopToVectorDim) {
    (void)localDim;
    if (loop == scalarLoopOp) {
      continue;
    }
    auto pit = ctx.parent->allLoopToVectorDim.find(loop);
    if (pit == ctx.parent->allLoopToVectorDim.end()) {
      continue;
    }
    parentDimOrder.push_back(static_cast<int>(pit->second));
  }
  llvm::sort(parentDimOrder);
  return parentDimOrder;
}

/// Record valueDimOrder on parent for one reduction final result when type is NPUVector.
static void tryRecordParentValueDimOrder(LoopVectorizationCtx &ctx, Value finalResult, npuvector::NPUVectorType nvt) {
  if (isMergedReductionLoop(ctx) && !isMergedReductionRoot(ctx) &&
      static_cast<int64_t>(ctx.getRank()) == nvt.getRank()) {
    ctx.parent->valueDimOrder[finalResult] = buildAxisOrder(static_cast<unsigned>(nvt.getRank()));
    return;
  }
  if (nvt.getRank() <= 0) {
    return;
  }
  SmallVector<int> parentDimOrder = buildParentDimOrderForReductionResult(ctx);
  if (static_cast<int64_t>(parentDimOrder.size()) == nvt.getRank()) {
    ctx.parent->valueDimOrder[finalResult] = parentDimOrder;
  }
}

static void finalizeReductionXOutputsWithParent(LoopVectorizationCtx &ctx, ArrayRef<Value> finalResults) {
  for (auto [scalarResult, finalResult] : llvm::zip(ctx.scalarLoop.getResults(), finalResults)) {
    ctx.parent->valueMapping.map(scalarResult, finalResult);
    if (ctx.parent->valueMapping.lookupOrNull(finalResult) == nullptr) {
      ctx.parent->valueMapping.map(finalResult, finalResult);
    }
    auto nvt = mlir::dyn_cast<npuvector::NPUVectorType>(finalResult.getType());
    if (nvt == nullptr) {
      continue;
    }
    tryRecordParentValueDimOrder(ctx, finalResult, nvt);
  }
}

static void finalizeReductionXOutputsTopLevel(LoopVectorizationCtx &ctx, ArrayRef<Value> finalResults) {
  for (auto [scalarResult, finalResult] : llvm::zip(ctx.scalarLoop.getResults(), finalResults)) {
    if (!scalarResult.use_empty()) {
      scalarResult.replaceAllUsesWith(finalResult);
    }
  }
  ctx.scalarLoop.erase();
}

static void finalizeReductionXOutputs(LoopVectorizationCtx &ctx, ArrayRef<Value> finalResults) {
  if (ctx.parent != nullptr) {
    finalizeReductionXOutputsWithParent(ctx, finalResults);
    return;
  }
  finalizeReductionXOutputsTopLevel(ctx, finalResults);
}

static LogicalResult collectReductionXYieldValues(LoopVectorizationCtx &ctx, scf::YieldOp scalarYield,
                                                  SmallVectorImpl<Value> &vecYieldVals) {
  Block *body = ctx.vecLoop.getBody();
  if (!body->empty() && isa<scf::YieldOp>(body->back())) {
    body->back().erase();
  }
  ctx.builder.setInsertionPointToEnd(body);

  vecYieldVals.reserve(scalarYield.getNumOperands());
  for (auto [idx, scalarVal] : llvm::enumerate(scalarYield.getOperands())) {
    Value mapped = ctx.valueMapping.lookupOrNull(scalarVal);
    if (!mapped) {
      return failure();
    }
    auto sliceIt = ctx.iterArgTileSlices.find(static_cast<unsigned>(idx));
    if (sliceIt != ctx.iterArgTileSlices.end()) {
      mapped = insertCurrentTileSlice(mapped, ctx, sliceIt->second, mapped.getLoc());
    }
    vecYieldVals.push_back(mapped);
  }
  return success();
}

static void recordReductionXResultDimOrders(LoopVectorizationCtx &ctx, ArrayRef<Value> vecYieldVals) {
  for (auto [vecResult, yieldVal] : llvm::zip(ctx.vecLoop.getResults(), vecYieldVals)) {
    auto outNvt = mlir::dyn_cast<npuvector::NPUVectorType>(vecResult.getType());
    auto ordIter = ctx.valueDimOrder.find(yieldVal);
    if (outNvt && ordIter != ctx.valueDimOrder.end() &&
        ordIter->second.size() == static_cast<unsigned>(outNvt.getRank())) {
      ctx.valueDimOrder[vecResult] = ordIter->second;
    }
  }
}

static LogicalResult reductionXVectorize(LoopVectorizationCtx &ctx) {
  if (ctx.parent == nullptr) {
    ctx.builder.setInsertionPoint(ctx.scalarLoop);
  }
  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) {
    return failure();
  }
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }

  Location loc = ctx.vecLoop.getLoc();
  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  const unsigned numResults = scalarYield.getNumOperands();
  if (numResults == 0 || numResults != ctx.vecLoop.getNumResults() || numResults != ctx.reductionKinds.size()) {
    return failure();
  }

  SmallVector<Value> vecYieldVals;
  if (failed(collectReductionXYieldValues(ctx, scalarYield, vecYieldVals))) {
    return failure();
  }
  ctx.builder.create<scf::YieldOp>(loc, vecYieldVals);
  recordReductionXResultDimOrders(ctx, vecYieldVals);

  ctx.builder.setInsertionPointAfter(ctx.vecLoop);

  SmallVector<Value> finalResults;
  finalResults.reserve(numResults);
  if (isMergedReductionLoop(ctx) && !isMergedReductionRoot(ctx)) {
    std::copy(ctx.vecLoop.getResults().begin(), ctx.vecLoop.getResults().end(), std::back_inserter(finalResults));
    finalizeReductionXOutputs(ctx, finalResults);
    return success();
  }

  for (unsigned idx = 0; idx < numResults; ++idx) {
    finalResults.push_back(buildReductionXLaneFinalResult(ctx, loc, idx));
  }

  finalizeReductionXOutputs(ctx, finalResults);

  return success();
}

static LogicalResult reductionYVectorize(LoopVectorizationCtx &ctx) {
  const bool innerIvIsScalar = ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
  if (!innerIvIsScalar) {
    ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;
  }

  if (ctx.vectorizationAxis) {
    Value outerIVMapping = ctx.valueMapping.lookupOrDefault(ctx.vectorizationAxis);
    if (outerIVMapping && outerIVMapping != ctx.vectorizationAxis) {
      ctx.valueMapping.map(ctx.vectorizationAxis, outerIVMapping);
    }
  }

  if (ctx.builder.getInsertionBlock() == nullptr) {
    ctx.builder.setInsertionPoint(ctx.scalarLoop);
  }

  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) {
    return failure();
  }
  if (failed(vectorizeLoopBody(ctx))) {
    ctx.vecLoop.erase();
    return failure();
  }
  finalizeReductionY(ctx);
  return success();
}

static LogicalResult broadcastVectorize(LoopVectorizationCtx &ctx) {
  const bool innerIvIsScalar = ctx.vectorizationAxis && ctx.vectorizationAxis != ctx.scalarLoop.getInductionVar();
  if (!innerIvIsScalar) {
    ctx.allLoopToVectorDim[ctx.scalarLoop] = 0;
  }

  if (ctx.vectorizationAxis) {
    Value outerIVMapping = ctx.valueMapping.lookupOrDefault(ctx.vectorizationAxis);
    if (outerIVMapping && outerIVMapping != ctx.vectorizationAxis) {
      ctx.valueMapping.map(ctx.vectorizationAxis, outerIVMapping);
    }
  }

  if (ctx.builder.getInsertionBlock() == nullptr) {
    ctx.builder.setInsertionPoint(ctx.scalarLoop);
  }

  ctx.vecLoop = createEmptyVectorizedLoop(ctx);
  if (!ctx.vecLoop) {
    return failure();
  }
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

static LoopVectorizationCtx createVF1SweepCtx(OpBuilder &builder) {
  // Phase 2 rank-0 sweep: empty allVectorSizes so getVectorType/getRank yield !npuvector.T (0-D),
  // not !npuvector<1xT> from actualStep=1.
  LoopVectorizationCtx ctx(LoopVectorizationInit{builder, SmallVector<int64_t>{}, SmallVector<Value>{},
                                                 SmallVector<Value>{}, VectorizationMode::Elementwise, scf::ForOp()});
  ctx.localDim = 0;
  ctx.vf1FuncLevelNoAnchor = true;
  return ctx;
}

struct VF1LoopLane {
  scf::ForOp loop;
  unsigned index;
  npuvector::NPUVectorType type;
};

static LogicalResult recordVF1LoopLane(scf::ForOp loop, unsigned index, SmallVectorImpl<VF1LoopLane> &loopLanes,
                                       SmallVectorImpl<Value> &valuesToVisit) {
  if (loop == nullptr || index >= loop.getNumRegionIterArgs() || index >= loop.getNumResults()) {
    return failure();
  }

  Type laneType = loop.getRegionIterArgs()[index].getType();
  npuvector::NPUVectorType vecType;
  if (auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(laneType)) {
    if (npuVecType.getRank() != 0) {
      return failure();
    }
    vecType = npuVecType;
  } else {
    if (laneType.isIndex() || mlir::isa<ShapedType>(laneType)) {
      return failure();
    }
    vecType = npuvector::NPUVectorType::get({}, laneType);
  }

  if (llvm::none_of(loopLanes,
                    [loop, index](const VF1LoopLane &lane) { return lane.loop == loop && lane.index == index; })) {
    loopLanes.push_back({loop, index, vecType});
  }
  valuesToVisit.push_back(loop.getRegionIterArgs()[index]);
  valuesToVisit.push_back(loop.getResult(index));
  return success();
}

static LogicalResult followVF1YieldLane(scf::YieldOp yieldOp, Value value, SmallVectorImpl<VF1LoopLane> &loopLanes,
                                        SmallVectorImpl<Value> &valuesToVisit) {
  auto loop = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
  if (loop == nullptr) {
    return success();
  }
  for (auto [index, operand] : llvm::enumerate(yieldOp.getOperands())) {
    if (operand == value && failed(recordVF1LoopLane(loop, index, loopLanes, valuesToVisit))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult followVF1ForInitLane(scf::ForOp loop, Value value, SmallVectorImpl<VF1LoopLane> &loopLanes,
                                          SmallVectorImpl<Value> &valuesToVisit) {
  bool found = false;
  for (auto [index, init] : llvm::enumerate(loop.getInitArgs())) {
    if (init != value) {
      continue;
    }
    found = true;
    if (failed(recordVF1LoopLane(loop, index, loopLanes, valuesToVisit))) {
      return failure();
    }
  }
  return found ? success() : failure();
}

/// Walk users of one SSA value and enqueue VF1 closure work (yield/init lanes or ops).
static LogicalResult enqueueVF1ClosureValueUsers(Value value, SmallVectorImpl<VF1LoopLane> &loopLanes,
                                                 SmallVectorImpl<Value> &valuesToVisit,
                                                 SmallVectorImpl<Operation *> &opsToVisit) {
  for (Operation *user : value.getUsers()) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (failed(followVF1YieldLane(yieldOp, value, loopLanes, valuesToVisit))) {
        return failure();
      }
      continue;
    }
    if (auto loop = dyn_cast<scf::ForOp>(user)) {
      if (failed(followVF1ForInitLane(loop, value, loopLanes, valuesToVisit))) {
        return failure();
      }
      continue;
    }
    if (llvm::any_of(user->getResults(),
                     [](Value result) { return mlir::isa<npuvector::NPUVectorType>(result.getType()); })) {
      continue;
    }
    opsToVisit.push_back(user);
  }
  return success();
}

static LogicalResult expandVF1ForwardClosure(memref::LoadOp root, DenseSet<Operation *> &closure,
                                             SmallVectorImpl<VF1LoopLane> &loopLanes) {
  closure.clear();
  loopLanes.clear();
  DenseSet<Value> visitedValues;
  SmallVector<Operation *> opsToVisit{root};
  SmallVector<Value> valuesToVisit;
  while (!opsToVisit.empty() || !valuesToVisit.empty()) {
    if (!valuesToVisit.empty()) {
      Value value = valuesToVisit.pop_back_val();
      if (!visitedValues.insert(value).second) {
        continue;
      }
      if (failed(enqueueVF1ClosureValueUsers(value, loopLanes, valuesToVisit, opsToVisit))) {
        return failure();
      }
      continue;
    }

    Operation *op = opsToVisit.pop_back_val();
    if (!closure.insert(op).second) {
      continue;
    }
    valuesToVisit.append(op->result_begin(), op->result_end());
  }
  return success();
}

static LogicalResult promoteVF1LoopLanes(SmallVectorImpl<VF1LoopLane> &loopLanes,
                                         const DenseSet<Operation *> &closure) {
  for (VF1LoopLane &lane : loopLanes) {
    Value init = lane.loop.getInitArgs()[lane.index];
    auto initVecType = mlir::dyn_cast<npuvector::NPUVectorType>(init.getType());
    if (initVecType && initVecType != lane.type) {
      return failure();
    }
    if (!initVecType && init.getType() != lane.type.getElementType()) {
      return failure();
    }
  }

  for (VF1LoopLane &lane : loopLanes) {
    lane.loop.getRegionIterArgs()[lane.index].setType(lane.type);
    lane.loop.getResult(lane.index).setType(lane.type);
  }

  for (VF1LoopLane &lane : loopLanes) {
    Value init = lane.loop.getInitArgs()[lane.index];
    Operation *initDef = init.getDefiningOp();
    if (!mlir::isa<npuvector::NPUVectorType>(init.getType()) && (initDef == nullptr || !closure.contains(initDef))) {
      OpBuilder builder(lane.loop);
      init = builder.create<npuvector::BroadcastOp>(lane.loop.getLoc(), lane.type, init, ValueRange{}, ValueRange{},
                                                    builder.getDenseI64ArrayAttr({}));
    }
    lane.loop.getInitArgsMutable()[lane.index].set(init);
  }
  return success();
}

static LogicalResult topoSortVF1Closure(const DenseSet<Operation *> &closureSet,
                                        SmallVectorImpl<Operation *> &topoOut) {
  DenseMap<Operation *, unsigned> indegree;
  for (Operation *op : closureSet) {
    indegree[op] = 0;
  }
  for (Operation *op : closureSet) {
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if ((def != nullptr) && closureSet.contains(def)) {
        indegree[op]++;
      }
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
        if (!closureSet.contains(user)) {
          continue;
        }
        if (--indegree[user] == 0) {
          ready.push_back(user);
        }
      }
    }
  }

  if (topoOut.size() != closureSet.size()) {
    return failure();
  }
  return success();
}

enum class Vf1ChainPromotionResult { Promoted, Skipped, FatalError };

static Vf1ChainPromotionResult tryPromoteVF1Chain(memref::LoadOp rootLoad) {
  DenseSet<Operation *> closure;
  SmallVector<VF1LoopLane> loopLanes;
  if (failed(expandVF1ForwardClosure(rootLoad, closure, loopLanes))) {
    return Vf1ChainPromotionResult::Skipped;
  }

  SmallVector<Operation *> topo;
  if (failed(topoSortVF1Closure(closure, topo))) {
    return Vf1ChainPromotionResult::Skipped;
  }
  if (failed(promoteVF1LoopLanes(loopLanes, closure))) {
    return Vf1ChainPromotionResult::Skipped;
  }

  OpBuilder builder(rootLoad);
  LoopVectorizationCtx ctx = createVF1SweepCtx(builder);

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
    if (!vecVal) {
      return Vf1ChainPromotionResult::FatalError;
    }
    op->getResult(0).replaceAllUsesWith(vecVal);
    op->erase();
  }
  return Vf1ChainPromotionResult::Promoted;
}

static LogicalResult expandVF1BackwardClosure(Value root, DenseSet<Operation *> &closure, bool &hasNpuVectorSource) {
  closure.clear();
  hasNpuVectorSource = false;
  DenseSet<Value> visited;
  SmallVector<Value> stack{root};

  while (!stack.empty()) {
    Value value = stack.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }
    if (auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(value.getType())) {
      if (npuVecType.getRank() != 0) {
        return failure();
      }
      hasNpuVectorSource = true;
      continue;
    }

    Operation *def = value.getDefiningOp();
    if (def == nullptr || isa<arith::ConstantOp>(def)) {
      continue;
    }
    if (isa<memref::LoadOp>(def) || def->getNumRegions() != 0 || def->getNumResults() != kUnaryOpOperandCount ||
        def->hasAttr(kSkipVectorizeAttr)) {
      return failure();
    }

    StringRef dialectName = def->getDialect()->getNamespace();
    if (dialectName != "arith" && dialectName != "math") {
      return failure();
    }

    closure.insert(def);
    stack.append(def->operand_begin(), def->operand_end());
  }

  return success();
}

static Vf1ChainPromotionResult tryPromoteVF1StoreChain(memref::StoreOp storeOp) {
  if (auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(storeOp.getValue().getType())) {
    if (npuVecType.getRank() != 0) {
      return Vf1ChainPromotionResult::Skipped;
    }
    OpBuilder builder(storeOp);
    LoopVectorizationCtx ctx = createVF1SweepCtx(builder);
    vectorizeStore(storeOp, ctx);
    storeOp.erase();
    return Vf1ChainPromotionResult::Promoted;
  }

  DenseSet<Operation *> closure;
  bool hasNpuVectorSource = false;
  if (failed(expandVF1BackwardClosure(storeOp.getValue(), closure, hasNpuVectorSource)) || closure.empty() ||
      !hasNpuVectorSource) {
    return Vf1ChainPromotionResult::Skipped;
  }

  SmallVector<Operation *> topo;
  if (failed(topoSortVF1Closure(closure, topo))) {
    return Vf1ChainPromotionResult::Skipped;
  }

  OpBuilder builder(storeOp);
  LoopVectorizationCtx ctx = createVF1SweepCtx(builder);
  for (Operation *op : topo) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(op);
    Value vecVal = vectorizeArithOp(op, ctx);
    if (!vecVal) {
      return Vf1ChainPromotionResult::FatalError;
    }
    op->getResult(0).replaceAllUsesWith(vecVal);
    op->erase();
  }

  builder.setInsertionPoint(storeOp);
  vectorizeStore(storeOp, ctx);
  storeOp.erase();
  return Vf1ChainPromotionResult::Promoted;
}

static bool isArithOrMathOp(Operation *op) {
  if (op == nullptr || op->getDialect() == nullptr) {
    return false;
  }
  StringRef dialectName = op->getDialect()->getNamespace();
  return dialectName == "arith" || dialectName == "math";
}

static bool hasRankZeroNpuVectorOperand(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(operand.getType());
    if (npuVecType && npuVecType.getRank() == 0) {
      return true;
    }
  }
  return false;
}

static LogicalResult promoteVF1YieldUsers(Operation *op, Value vecVal) {
  auto vecType = mlir::dyn_cast<npuvector::NPUVectorType>(vecVal.getType());
  if (!vecType || vecType.getRank() != 0) {
    return success();
  }

  SmallVector<VF1LoopLane> loopLanes;
  SmallVector<Value> unusedValuesToVisit;
  for (Operation *user : op->getResult(0).getUsers()) {
    auto yieldOp = dyn_cast<scf::YieldOp>(user);
    if (!yieldOp) {
      continue;
    }
    if (failed(followVF1YieldLane(yieldOp, op->getResult(0), loopLanes, unusedValuesToVisit))) {
      return failure();
    }
  }
  if (loopLanes.empty()) {
    return success();
  }

  DenseSet<Operation *> closure;
  closure.insert(op);
  return promoteVF1LoopLanes(loopLanes, closure);
}

static Vf1ChainPromotionResult tryPromoteVF1ArithOp(Operation *op) {
  if (!isArithOrMathOp(op) || op->getNumRegions() != 0 || op->getNumResults() != kUnaryOpOperandCount ||
      op->getNumOperands() == 0 || hasIndexResult(*op) || op->hasAttr(kSkipVectorizeAttr) ||
      isa<arith::ConstantOp>(op)) {
    return Vf1ChainPromotionResult::Skipped;
  }
  if (mlir::isa<npuvector::NPUVectorType>(op->getResult(0).getType())) {
    return Vf1ChainPromotionResult::Skipped;
  }
  if (!hasRankZeroNpuVectorOperand(op)) {
    return Vf1ChainPromotionResult::Skipped;
  }

  OpBuilder builder(op);
  LoopVectorizationCtx ctx = createVF1SweepCtx(builder);
  if (failed(handleArithOrMathOp(*op, ctx))) {
    return Vf1ChainPromotionResult::FatalError;
  }
  Value vecVal = ctx.valueMapping.lookupOrNull(op->getResult(0));
  if (!vecVal || vecVal == op->getResult(0) || !mlir::isa<npuvector::NPUVectorType>(vecVal.getType())) {
    return Vf1ChainPromotionResult::Skipped;
  }

  if (failed(promoteVF1YieldUsers(op, vecVal))) {
    return Vf1ChainPromotionResult::FatalError;
  }

  op->getResult(0).replaceAllUsesWith(vecVal);
  op->erase();
  return Vf1ChainPromotionResult::Promoted;
}

static LogicalResult runArithVF1Sweep(func::FuncOp funcOp, bool &changed) {
  changed = false;
  SmallVector<Operation *> ops;
  funcOp.walk([&](Operation *op) {
    if (isArithOrMathOp(op)) {
      ops.push_back(op);
    }
  });

  for (Operation *op : ops) {
    if (op == nullptr || op->getBlock() == nullptr) {
      continue;
    }
    Vf1ChainPromotionResult outcome = tryPromoteVF1ArithOp(op);
    if (outcome == Vf1ChainPromotionResult::FatalError) {
      return failure();
    }
    if (outcome == Vf1ChainPromotionResult::Promoted) {
      changed = true;
    }
  }
  return success();
}

static LogicalResult runMemRefLoadVF1Sweep(func::FuncOp funcOp, bool &changed) {
  changed = false;
  SmallVector<memref::LoadOp> loads;
  funcOp.walk([&loads](memref::LoadOp ld) { loads.push_back(ld); });

  for (memref::LoadOp ld : loads) {
    if (ld == nullptr || ld->getBlock() == nullptr) {
      continue;
    }
    Vf1ChainPromotionResult outcome = tryPromoteVF1Chain(ld);
    if (outcome == Vf1ChainPromotionResult::FatalError) {
      return failure();
    }
    if (outcome == Vf1ChainPromotionResult::Promoted) {
      changed = true;
    }
  }
  return success();
}

static LogicalResult runMemRefStoreVF1Sweep(func::FuncOp funcOp, bool &changed) {
  changed = false;
  SmallVector<memref::StoreOp> stores;
  funcOp.walk([&stores](memref::StoreOp storeOp) { stores.push_back(storeOp); });

  for (memref::StoreOp storeOp : stores) {
    if (storeOp == nullptr || storeOp->getBlock() == nullptr) {
      continue;
    }
    Vf1ChainPromotionResult outcome = tryPromoteVF1StoreChain(storeOp);
    if (outcome == Vf1ChainPromotionResult::FatalError) {
      return failure();
    }
    if (outcome == Vf1ChainPromotionResult::Promoted) {
      changed = true;
    }
  }
  return success();
}

static LogicalResult runPhase2RankZeroSweep(func::FuncOp funcOp) {
  constexpr unsigned kMaxRounds = 64;
  for (unsigned round = 0; round < kMaxRounds; ++round) {
    bool loadChanged = false;
    bool storeChanged = false;
    bool arithChanged = false;
    if (failed(runMemRefLoadVF1Sweep(funcOp, loadChanged)) || failed(runMemRefStoreVF1Sweep(funcOp, storeChanged)) ||
        failed(runArithVF1Sweep(funcOp, arithChanged))) {
      return failure();
    }
    if (!arithChanged && !loadChanged && !storeChanged) {
      return success();
    }
  }
  return funcOp.emitError("npuvector-vectorize: Phase 2 rank-0 sweep did not converge");
}

static void runVectorization(scf::ForOp loop, VectorizationMode mode, OpBuilder &builder, int64_t maxStepFromAttr,
                             bool isDynamic) {
  OpBuilder attrBuilder(builder.getContext());
  attrBuilder.setInsertionPoint(loop);
  if (mode == VectorizationMode::ReductionX) {
    if (std::optional<MergedReductionGroup> group = tryBuildMergedReductionGroup(loop, attrBuilder)) {
      LoopVectorizationCtx ctx(LoopVectorizationInit{builder, group->vectorSizes, group->vectorSizeValues,
                                                     group->maxStepValues, mode, loop, loop.getInductionVar()});
      ctx.mergedReductionLoops.reserve(group->loops.size());
      for (auto [idx, groupLoop] : llvm::enumerate(group->loops)) {
        ctx.allLoopToVectorDim[groupLoop] = static_cast<unsigned>(idx);
        ctx.mergedReductionLoops.push_back(groupLoop.getOperation());
      }
      ctx.localDim = 0;
      processLoop(ctx);
      return;
    }
  }

  int64_t actualStep = computeStaticVectorSize(loop, maxStepFromAttr);
  Value maxStepValue = attrBuilder.create<arith::ConstantIndexOp>(loop.getLoc(), maxStepFromAttr);
  Value vectorSizeValue =
    isDynamic ? computeDynamicVectorSize(loop, maxStepValue, attrBuilder, loop.getLoc(), nullptr) : Value();

  SmallVector<int64_t> vecSizes = {actualStep};
  SmallVector<Value> vecSizeValues = {vectorSizeValue};
  SmallVector<Value> maxSteps = {maxStepValue};
  LoopVectorizationCtx ctx(LoopVectorizationInit{builder, std::move(vecSizes), std::move(vecSizeValues),
                                                 std::move(maxSteps), mode, loop, loop.getInductionVar()});

  const bool innerIvIsScalar = (mode == VectorizationMode::ReductionY || mode == VectorizationMode::Broadcast);
  if (!innerIvIsScalar) {
    ctx.allLoopToVectorDim[loop] = 0;
    ctx.localDim = 0;
  }
  processLoop(ctx);
}

class NPUVectorVectorize : public mlir::scf::impl::NPUVectorVectorizeBase<NPUVectorVectorize> {
 public:
  NPUVectorVectorize() = default;
  NPUVectorVectorize(const NPUVectorVectorize &) = default;
  NPUVectorVectorize &operator=(const NPUVectorVectorize &) = default;

  [[nodiscard]] StringRef getArgument() const override { return "npuvector-vectorize"; }

  [[nodiscard]] StringRef getDescription() const override {
    return "SCF loop vectorization using NPUVector with dynamic shape support";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, npuvector::NPUVectorDialect, memref::MemRefDialect, func::FuncDialect,
                    affine::AffineDialect, arith::ArithDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!hacc::utils::isDevice(funcOp)) {
      return;
    }

    OpBuilder builder(&getContext());

    SmallVector<scf::ForOp> allCandidateLoops;
    funcOp.walk([&allCandidateLoops](scf::ForOp forOp) {
      if (hasVectorizationAttr(forOp)) {
        allCandidateLoops.push_back(forOp);
      }
    });

    SmallVector<scf::ForOp> topLevelLoops;
    for (scf::ForOp loop : allCandidateLoops) {
      bool isNested = false;
      for (Operation *parent = loop->getParentOp(); (parent != nullptr) && !isa<func::FuncOp>(parent);
           parent = parent->getParentOp()) {
        if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
          if (hasVectorizationAttr(parentFor)) {
            isNested = true;
            break;
          }
        }
      }
      if (!isNested) {
        topLevelLoops.push_back(loop);
      }
    }

    for (scf::ForOp loop : topLevelLoops) {
      int64_t maxStepFromAttr = -1;
      VectorizationMode mode = getVectorizationMode(loop, maxStepFromAttr);
      if (mode == VectorizationMode::None || maxStepFromAttr <= 0) {
        continue;
      }

      auto [skip, isDynamic] = checkLoopEligibility(loop);
      if (skip) {
        continue;
      }

      runVectorization(loop, mode, builder, maxStepFromAttr, isDynamic);
    }

    if (failed(runPhase2RankZeroSweep(funcOp))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {
namespace scf {
std::unique_ptr<OperationPass<func::FuncOp>> createNPUVectorVectorizePass() {
  return std::make_unique<NPUVectorVectorize>();
}
}  // namespace scf
}  // namespace mlir
