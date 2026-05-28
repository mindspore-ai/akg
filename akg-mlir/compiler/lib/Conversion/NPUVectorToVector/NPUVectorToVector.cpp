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

#include "akg/Conversion/NPUVectorToVector/NPUVectorToVector.h"

#include <algorithm>
#include <iterator>
#include <optional>
#include <type_traits>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APInt.h"

namespace mlir {
#define GEN_PASS_DEF_NPUVECTORTOVECTOR
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

using namespace mlir;  // NOLINT(build/namespaces)

namespace {

VectorType toVectorType(npuvector::NPUVectorType t) { return VectorType::get(t.getShape(), t.getElementType()); }

std::optional<int64_t> getConstantIndexValue(Value v) {
  if (auto c = v.getDefiningOp<arith::ConstantIndexOp>()) return c.value();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (isa<IndexType>(c.getType()))
      if (auto intAttr = dyn_cast<IntegerAttr>(c.getValue())) return intAttr.getInt();
  }
  return std::nullopt;
}

std::optional<VectorType> tryResolveStaticVectorType(npuvector::NPUVectorType npu, ValueRange maxSizes) {
  SmallVector<int64_t, 8> shape;
  unsigned midx = 0;
  for (int64_t d : npu.getShape()) {
    if (d != ShapedType::kDynamic) {
      shape.push_back(d);
      continue;
    }
    if (midx >= maxSizes.size()) return std::nullopt;
    auto c = getConstantIndexValue(maxSizes[midx++]);
    if (!c) return std::nullopt;
    shape.push_back(*c);
  }
  return VectorType::get(shape, npu.getElementType());
}

VectorType inferStaticResultVecFromOperands(npuvector::NPUVectorType outNpu, TypeRange operandTypes) {
  VectorType fromNpu = toVectorType(outNpu);
  if (fromNpu.hasStaticShape()) return fromNpu;
  Type elemTy = outNpu.getElementType();
  for (Type t : operandTypes) {
    auto vt = dyn_cast<VectorType>(t);
    if (vt && vt.hasStaticShape() && vt.getRank() == fromNpu.getRank() && vt.getElementType() == elemTy)
      return VectorType::get(vt.getShape(), elemTy);
  }
  for (Type t : operandTypes) {
    auto vt = dyn_cast<VectorType>(t);
    if (vt && vt.hasStaticShape() && vt.getRank() == fromNpu.getRank()) return VectorType::get(vt.getShape(), elemTy);
  }
  return fromNpu;
}

VectorType maskTypeFor(VectorType vecTy, OpBuilder &b) { return VectorType::get(vecTy.getShape(), b.getI1Type()); }

Value buildBoundedMask(PatternRewriter &rewriter, Location loc, VectorType staticVecTy,
                       npuvector::NPUVectorType origNpu, ValueRange dynamicSizes) {
  SmallVector<Value, 4> maskOperands;
  unsigned ds = 0;
  for (int64_t i = 0; i < origNpu.getRank(); ++i) {
    if (origNpu.getShape()[i] != ShapedType::kDynamic)
      maskOperands.push_back(rewriter.create<arith::ConstantIndexOp>(loc, staticVecTy.getDimSize(i)));
    else
      maskOperands.push_back(dynamicSizes[ds++]);
  }
  return rewriter.create<vector::CreateMaskOp>(loc, maskTypeFor(staticVecTy, rewriter), maskOperands).getResult();
}

Value combineMasks(PatternRewriter &rewriter, Location loc, Value a, Value b) {
  if (!a) return b;
  if (!b) return a;
  return rewriter.create<arith::AndIOp>(loc, a, b).getResult();
}

bool sameIndexList(ValueRange a, ValueRange b) {
  if (a.size() != b.size()) return false;
  for (auto [x, y] : llvm::zip(a, b))
    if (x != y) return false;
  return true;
}

/// Walk predecessors in the same block to find a tile load with the same source memref and indices as the store.
struct TileMeta {
  VectorType vecTy;
  /// Synthetic bounded/full mask from dynamic tiles; empty when shape is fully static and no inferred mask.
  Value tileMask;
};

std::optional<TileMeta> findTileMetaForWrite(npuvector::TransferWriteOp writeOp, PatternRewriter &rewriter) {
  ValueRange wIdx = writeOp.getIndices();
  Value writeSrc = writeOp.getSource();
  Location loc = writeOp.getLoc();
  for (Operation *walk = writeOp.getOperation()->getPrevNode(); walk; walk = walk->getPrevNode()) {
    if (auto nread = dyn_cast<npuvector::TransferReadOp>(walk)) {
      if (nread.getSource() != writeSrc) continue;
      if (!sameIndexList(nread.getIndices(), wIdx)) continue;
      auto npuRes = dyn_cast<npuvector::NPUVectorType>(nread.getVector().getType());
      if (!npuRes) continue;
      auto vecTy = tryResolveStaticVectorType(npuRes, nread.getMaxSizes());
      if (!vecTy) continue;
      bool hasDyn = llvm::is_contained(npuRes.getShape(), ShapedType::kDynamic);
      Value tileMask = hasDyn ? buildBoundedMask(rewriter, loc, *vecTy, npuRes, nread.getDynamicSizes()) : Value();
      tileMask = combineMasks(rewriter, loc, nread.getMask(), tileMask);
      return TileMeta{*vecTy, tileMask};
    }
    if (auto vread = dyn_cast<vector::TransferReadOp>(walk)) {
      if (vread.getSource() != writeSrc) continue;
      if (!sameIndexList(vread.getIndices(), wIdx)) continue;
      VectorType vt = vread.getVectorType();
      if (!vt.hasStaticShape()) continue;
      Value m = vread.getMask();
      return TileMeta{vt, m};
    }
  }
  return std::nullopt;
}

/// When no same-indices transfer_read appears before the write, infer static vector type and
/// tile mask from the stored value's definition (broadcast/read metadata or elementwise shape).
static std::optional<TileMeta> inferTileMetaFromStoredValue(Value vecValue, Location loc, PatternRewriter &rewriter) {
  Type ty = vecValue.getType();
  if (auto vt = dyn_cast<VectorType>(ty)) {
    if (!vt.hasStaticShape()) return std::nullopt;
    return TileMeta{vt, Value()};
  }
  auto npuTy = dyn_cast<npuvector::NPUVectorType>(ty);
  if (!npuTy) return std::nullopt;
  if (!llvm::is_contained(npuTy.getShape(), ShapedType::kDynamic)) {
    VectorType vt = toVectorType(npuTy);
    return TileMeta{vt, Value()};
  }
  Operation *def = vecValue.getDefiningOp();
  if (!def) return std::nullopt;

  if (auto br = dyn_cast<npuvector::BroadcastOp>(def)) {
    auto resNpu = dyn_cast<npuvector::NPUVectorType>(br.getResult().getType());
    if (!resNpu) return std::nullopt;
    auto vecTy = tryResolveStaticVectorType(resNpu, br.getMaxSizes());
    if (!vecTy) return std::nullopt;
    bool hasDyn = llvm::is_contained(resNpu.getShape(), ShapedType::kDynamic);
    if (hasDyn && br.getDynamicSizes().empty()) return std::nullopt;
    Value tileMask = hasDyn ? buildBoundedMask(rewriter, loc, *vecTy, resNpu, br.getDynamicSizes()) : Value();
    return TileMeta{*vecTy, tileMask};
  }
  if (auto rd = dyn_cast<npuvector::TransferReadOp>(def)) {
    auto resNpu = dyn_cast<npuvector::NPUVectorType>(rd.getVector().getType());
    if (!resNpu) return std::nullopt;
    auto vecTy = tryResolveStaticVectorType(resNpu, rd.getMaxSizes());
    if (!vecTy) return std::nullopt;
    bool hasDyn = llvm::is_contained(resNpu.getShape(), ShapedType::kDynamic);
    if (hasDyn && rd.getDynamicSizes().empty()) return std::nullopt;
    Value tileMask = hasDyn ? buildBoundedMask(rewriter, loc, *vecTy, resNpu, rd.getDynamicSizes()) : Value();
    tileMask = combineMasks(rewriter, loc, rd.getMask(), tileMask);
    return TileMeta{*vecTy, tileMask};
  }

  VectorType inferred = inferStaticResultVecFromOperands(npuTy, TypeRange(def->getOperands()));
  if (!inferred.hasStaticShape()) return std::nullopt;
  return TileMeta{inferred, Value()};
}

static unsigned getEffectiveVectorRankForXferOp(ShapedType shapedType, VectorType vectorType) {
  unsigned elementVectorRank = 0;
  if (auto elementVectorType = dyn_cast<VectorType>(shapedType.getElementType()))
    elementVectorRank += elementVectorType.getRank();
  return vectorType.getRank() - elementVectorRank;
}

/// Aligns with MLIR's vector::getTransferMinorIdentityMap: 0-D base + vector<1xt> needs a map with
/// one result (constant 0), not affine_map<() -> ()> or getMinorIdentityMap(0, rank).
static AffineMap getXferMinorIdentityMap(ShapedType shapedType, VectorType vectorType) {
  MLIRContext *ctx = shapedType.getContext();
  if (shapedType.getRank() == 0 && vectorType.getShape() == ArrayRef<int64_t>{1})
    return AffineMap::get(/*numDims=*/0, /*numSymbols=*/0, ArrayRef<AffineExpr>{getAffineConstantExpr(0, ctx)}, ctx);
  return AffineMap::getMinorIdentityMap(shapedType.getRank(), getEffectiveVectorRankForXferOp(shapedType, vectorType),
                                        ctx);
}

AffineMap getTransferPermutation(Operation *op, ShapedType baseShaped, VectorType vectorType) {
  if (auto attr = op->getAttrOfType<AffineMapAttr>("permutation_map")) {
    AffineMap m = attr.getValue();
    if (m.getNumResults() == vectorType.getRank()) return m;
  }
  return getXferMinorIdentityMap(baseShaped, vectorType);
}

ArrayAttr getInBoundsArrayAttr(PatternRewriter &rewriter, Operation *op, unsigned vectorRank) {
  if (auto arr = op->getAttrOfType<ArrayAttr>("in_bounds")) return arr;
  // Older MLIR may not have BoolArrayAttr; only ArrayAttr of bool/i1 is supported here.
  return rewriter.getArrayAttr(SmallVector<Attribute>(vectorRank, rewriter.getBoolAttr(false)));
}

struct NpuTransferReadToVector : public OpRewritePattern<npuvector::TransferReadOp> {
  using OpRewritePattern<npuvector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npuvector::TransferReadOp op, PatternRewriter &rewriter) const override {
    auto npuTy = dyn_cast<npuvector::NPUVectorType>(op.getVector().getType());
    if (!npuTy) return failure();
    auto vecTy = tryResolveStaticVectorType(npuTy, op.getMaxSizes());
    if (!vecTy) return failure();

    bool hasDyn = llvm::is_contained(npuTy.getShape(), ShapedType::kDynamic);
    if (hasDyn && op.getDynamicSizes().empty()) return failure();

    Location loc = op.getLoc();
    Value boundedMask = hasDyn ? buildBoundedMask(rewriter, loc, *vecTy, npuTy, op.getDynamicSizes()) : Value();
    Value mask = combineMasks(rewriter, loc, op.getMask(), boundedMask);

    auto baseShaped = cast<ShapedType>(op.getSource().getType());
    AffineMap perm = getTransferPermutation(op.getOperation(), baseShaped, *vecTy);
    auto inBounds = getInBoundsArrayAttr(rewriter, op.getOperation(), vecTy->getRank());

    auto newOp = rewriter.create<vector::TransferReadOp>(loc, *vecTy, op.getSource(), op.getIndices(), perm,
                                                         op.getPadding(), mask, inBounds);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct NpuTransferWriteToVector : public OpRewritePattern<npuvector::TransferWriteOp> {
  using OpRewritePattern<npuvector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npuvector::TransferWriteOp op, PatternRewriter &rewriter) const override {
    Value vecValue = op->getOperand(0);
    Type vecOpTy = vecValue.getType();

    if (!isa<npuvector::NPUVectorType, VectorType>(vecOpTy)) return failure();

    std::optional<TileMeta> meta = findTileMetaForWrite(op, rewriter);
    if (!meta) meta = inferTileMetaFromStoredValue(vecValue, op.getLoc(), rewriter);
    if (!meta) return failure();
    VectorType staticTy = meta->vecTy;
    Value tileMask = meta->tileMask;

    if (!isa<VectorType>(vecOpTy) || cast<VectorType>(vecOpTy) != staticTy) return failure();

    auto baseShaped = cast<ShapedType>(op.getSource().getType());
    Location loc = op.getLoc();
    Value mask = combineMasks(rewriter, loc, op.getMask(), tileMask);

    // transfer_write cannot use broadcast permutation maps; 0-D base + vector<1xt> only
    // admits broadcast maps. Lift base to memref<1xt> with expand_shape so (d0)->(d0) applies.
    if (auto memTy = dyn_cast<MemRefType>(op.getSource().getType())) {
      if (memTy.getRank() == 0 && staticTy.getRank() == 1 && staticTy.getDimSize(0) == 1 && !staticTy.isScalable() &&
          staticTy.getElementType() == memTy.getElementType()) {
        auto expandedTy = MemRefType::get({1}, memTy.getElementType());
        Value base1 =
          rewriter.create<memref::ExpandShapeOp>(loc, expandedTy, op.getSource(), SmallVector<ReassociationIndices>())
            .getResult();
        Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        auto rank1Base = cast<ShapedType>(base1.getType());
        AffineMap perm = getTransferPermutation(op.getOperation(), rank1Base, staticTy);
        auto inBounds = getInBoundsArrayAttr(rewriter, op.getOperation(), staticTy.getRank());
        auto newWrite = rewriter.create<vector::TransferWriteOp>(loc, vecValue, base1, ValueRange{c0},
                                                                 AffineMapAttr::get(perm), mask, inBounds);
        if (op.getNumResults() != 0)
          rewriter.replaceOp(op, newWrite->getResults());
        else
          rewriter.eraseOp(op);
        return success();
      }
    }

    AffineMap perm = getTransferPermutation(op.getOperation(), baseShaped, staticTy);
    auto inBounds = getInBoundsArrayAttr(rewriter, op.getOperation(), staticTy.getRank());

    auto newWrite = rewriter.create<vector::TransferWriteOp>(loc, vecValue, op.getSource(), op.getIndices(),
                                                             AffineMapAttr::get(perm), mask, inBounds);
    if (op.getNumResults() != 0)
      rewriter.replaceOp(op, newWrite->getResults());
    else
      rewriter.eraseOp(op);
    return success();
  }
};

constexpr int64_t kF32VectorChunkSize = 64;
constexpr int64_t kF16VectorChunkSize = 128;

std::optional<int64_t> get1DVectorChunkSize(Type elemTy) {
  if (elemTy.isF32()) return kF32VectorChunkSize;
  if (elemTy.isF16()) return kF16VectorChunkSize;
  return std::nullopt;
}

struct VectorChunkConfig {
  int64_t chunkSize;
  int64_t totalSize;
  VectorType chunkTy;
};

std::optional<VectorChunkConfig> get1DVectorChunkConfig(VectorType vecTy) {
  if (!vecTy) return std::nullopt;
  if (vecTy.getRank() != 1 || !vecTy.hasStaticShape()) return std::nullopt;
  std::optional<int64_t> chunkSizeOpt = get1DVectorChunkSize(vecTy.getElementType());
  if (!chunkSizeOpt) return std::nullopt;
  int64_t chunkSize = *chunkSizeOpt;
  int64_t totalSize = vecTy.getDimSize(0);
  if (totalSize <= chunkSize) return std::nullopt;
  if (totalSize % chunkSize != 0) return std::nullopt;
  Type elemTy = vecTy.getElementType();
  return VectorChunkConfig{chunkSize, totalSize, VectorType::get({chunkSize}, elemTy)};
}

Value getCombiningNeutral(OpBuilder &builder, Location loc, vector::CombiningKind kind, Type elemTy) {
  const bool isMul = kind == vector::CombiningKind::MUL;
  if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
    return builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(floatTy, isMul ? 1.0 : 0.0));
  }
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned w = intTy.getWidth();
    return builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, isMul ? APInt(w, 1) : APInt(w, 0)));
  }
  if (isa<IndexType>(elemTy)) return builder.create<arith::ConstantIndexOp>(loc, isMul ? 1 : 0);
  return builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elemTy));
}

SmallVector<Value> offsetMemrefIndices(OpBuilder &builder, Location loc, ValueRange baseIndices, Value elementOffset) {
  if (baseIndices.empty()) return {elementOffset};
  SmallVector<Value> indices(baseIndices.begin(), baseIndices.end());
  indices[0] = builder.create<arith::AddIOp>(loc, indices[0], elementOffset).getResult();
  return indices;
}

vector::TransferReadOp getDefiningVectorTransferRead(Value v) { return v.getDefiningOp<vector::TransferReadOp>(); }

ArrayAttr getAllTrueInBoundsAttr(OpBuilder &builder, unsigned vectorRank) {
  return builder.getArrayAttr(SmallVector<Attribute>(vectorRank, builder.getBoolAttr(true)));
}

Value buildVectorChunkNeutral(OpBuilder &builder, Location loc, VectorType chunkTy, vector::CombiningKind kind) {
  Type elemTy = chunkTy.getElementType();
  Attribute neutral;
  if (kind == vector::CombiningKind::MUL) {
    if (auto floatTy = dyn_cast<FloatType>(elemTy))
      neutral = builder.getFloatAttr(floatTy, 1.0);
    else if (auto intTy = dyn_cast<IntegerType>(elemTy))
      neutral = builder.getIntegerAttr(intTy, APInt(intTy.getWidth(), 1));
    else
      neutral = builder.getZeroAttr(elemTy);
  } else {
    neutral = builder.getZeroAttr(elemTy);
  }
  return builder.create<arith::ConstantOp>(loc, chunkTy, DenseElementsAttr::get(chunkTy, neutral));
}

Value buildChunkTransferRead(OpBuilder &builder, Location loc, vector::TransferReadOp readOp, Value elementOffset,
                             VectorType chunkTy, ArrayAttr inBounds = {}) {
  auto indices = offsetMemrefIndices(builder, loc, readOp.getIndices(), elementOffset);
  ArrayAttr bounds = inBounds ? inBounds : readOp.getInBoundsAttr();
  return builder.create<vector::TransferReadOp>(loc, chunkTy, readOp.getSource(), indices, readOp.getPermutationMap(),
                                                readOp.getPadding(), readOp.getMask(), bounds);
}

void buildChunkTransferWrite(OpBuilder &builder, Location loc, vector::TransferWriteOp writeOp, Value elementOffset,
                             Value chunk) {
  auto indices = offsetMemrefIndices(builder, loc, writeOp.getIndices(), elementOffset);
  builder.create<vector::TransferWriteOp>(loc, chunk, writeOp.getSource(), indices, writeOp.getPermutationMapAttr(),
                                          writeOp.getMask(), writeOp.getInBoundsAttr());
}

std::optional<Value> getSplatChunkConstant(OpBuilder &builder, Location loc, Value vector,
                                           const VectorChunkConfig &cfg) {
  auto cst = vector.getDefiningOp<arith::ConstantOp>();
  if (!cst) return std::nullopt;
  auto dense = dyn_cast<DenseElementsAttr>(cst.getValue());
  if (!dense || !dense.isSplat()) return std::nullopt;
  return builder.create<arith::ConstantOp>(loc, cfg.chunkTy, dense.resizeSplat(cfg.chunkTy)).getResult();
}

/// Fuse transfer_read(s) + elementwise + transfer_write on oversized 1-D vectors into one
/// scf.for with direct memref chunked transfer_read/write (64xf32 / 128xf16).
struct FuseChunkedMemrefElementwise : public RewritePattern {
  explicit FuseChunkedMemrefElementwise(MLIRContext *ctx) : RewritePattern(Pattern::MatchAnyOpTypeTag{}, 3, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (!isa<arith::ArithDialect>(op->getDialect()) || isa<arith::ConstantOp>(op)) return failure();
    if (op->getNumRegions() != 0 || op->getNumResults() != 1 || !op->hasOneUse()) return failure();

    auto outTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!outTy) return failure();
    auto cfgOpt = get1DVectorChunkConfig(outTy);
    if (!cfgOpt) return failure();
    const VectorChunkConfig &cfg = *cfgOpt;

    auto writeOp = dyn_cast<vector::TransferWriteOp>(*op->user_begin());
    if (!writeOp || writeOp.getVector() != op->getResult(0)) return failure();

    SmallVector<vector::TransferReadOp, 4> reads;
    reads.reserve(op->getNumOperands());
    for (Value operand : op->getOperands()) {
      auto readOp = getDefiningVectorTransferRead(operand);
      if (!readOp || readOp.getVectorType() != outTy) return failure();
      reads.push_back(readOp);
    }

    Location loc = op->getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, cfg.totalSize);
    Value cChunk = rewriter.create<arith::ConstantIndexOp>(loc, cfg.chunkSize);
    OperationName opName = op->getName();
    auto attrs = op->getAttrs();

    rewriter.setInsertionPoint(reads.front());
    auto forOp = rewriter.create<scf::ForOp>(
      loc, c0, cTotal, cChunk, std::nullopt, [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
        SmallVector<Value, 4> chunkOperands;
        chunkOperands.reserve(reads.size());
        std::transform(reads.begin(), reads.end(), std::back_inserter(chunkOperands),
                       [&](vector::TransferReadOp readOp) {
                         return buildChunkTransferRead(b, bodyLoc, readOp, iv, cfg.chunkTy);
                       });

        OperationState st(bodyLoc, opName);
        st.addOperands(chunkOperands);
        st.addTypes(cfg.chunkTy);
        st.addAttributes(attrs);
        Value chunkResult = b.create(st)->getResult(0);
        buildChunkTransferWrite(b, bodyLoc, writeOp, iv, chunkResult);
        b.create<scf::YieldOp>(bodyLoc);
      });

    (void)forOp;
    rewriter.eraseOp(writeOp);
    rewriter.eraseOp(op);
    for (vector::TransferReadOp readOp : reads)
      if (readOp->use_empty()) rewriter.eraseOp(readOp);
    return success();
  }
};

Value buildReduction(OpBuilder &builder, Location loc, vector::CombiningKind kind, VectorType srcVec, Value v,
                     Value acc) {
  Type elemTy = srcVec.getElementType();
  Value accIn = acc;
  if (!accIn) {
    const bool isMul = kind == vector::CombiningKind::MUL;
    if (auto floatTy = dyn_cast<FloatType>(elemTy)) {
      accIn = builder.create<arith::ConstantOp>(loc, builder.getFloatAttr(floatTy, isMul ? 1.0 : 0.0));
    } else if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
      unsigned w = intTy.getWidth();
      accIn = builder.create<arith::ConstantOp>(loc, builder.getIntegerAttr(intTy, isMul ? APInt(w, 1) : APInt(w, 0)));
    } else if (isa<IndexType>(elemTy)) {
      accIn = builder.create<arith::ConstantIndexOp>(loc, isMul ? 1 : 0);
    } else {
      accIn = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elemTy));
    }
  }
  SmallVector<int64_t, 8> dims;
  for (int64_t d = 0, r = srcVec.getRank(); d < r; ++d) dims.push_back(d);
  SmallVector<Attribute, 8> dimAttrs;
  dimAttrs.reserve(dims.size());
  std::transform(dims.begin(), dims.end(), std::back_inserter(dimAttrs),
                 [&](int64_t d) { return builder.getIntegerAttr(builder.getI64Type(), d); });
  return builder.create<vector::MultiDimReductionOp>(loc, elemTy, kind, v, accIn, builder.getArrayAttr(dimAttrs));
}

/// Fuse large 1-D reduction + broadcast-to-1 + memref store: chunked vector accumulate in scf.for,
/// then vector.multi_reduction outside the loop with seed from the store memref (extractelement).
struct FuseChunkedReductionMemrefStore : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  explicit FuseChunkedReductionMemrefStore(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/3) {}

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op, PatternRewriter &rewriter) const override {
    if (op.getKind() != vector::CombiningKind::ADD) return failure();
    if (!op->hasOneUse()) return failure();

    auto srcTy = dyn_cast<VectorType>(op.getSource().getType());
    if (!srcTy) return failure();
    auto cfgOpt = get1DVectorChunkConfig(srcTy);
    if (!cfgOpt) return failure();
    const VectorChunkConfig &cfg = *cfgOpt;

    auto readOp = getDefiningVectorTransferRead(op.getSource());
    if (!readOp) return failure();

    auto bcastOp = dyn_cast<vector::BroadcastOp>(*op->user_begin());
    if (!bcastOp || !bcastOp->hasOneUse()) return failure();
    auto outVecTy = dyn_cast<VectorType>(bcastOp.getResult().getType());
    if (!outVecTy || outVecTy.getNumElements() != 1) return failure();

    auto writeOp = dyn_cast<vector::TransferWriteOp>(*bcastOp->user_begin());
    if (!writeOp || writeOp.getVector() != bcastOp.getResult()) return failure();

    Location loc = op.getLoc();
    Type elemTy = srcTy.getElementType();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, cfg.totalSize);
    Value cChunk = rewriter.create<arith::ConstantIndexOp>(loc, cfg.chunkSize);
    Value pad = readOp.getPadding();
    ArrayAttr inBoundsTrue = getAllTrueInBoundsAttr(rewriter, cfg.chunkTy.getRank());
    ArrayAttr inBoundsWriteTrue = getAllTrueInBoundsAttr(rewriter, outVecTy.getRank());

    rewriter.setInsertionPoint(readOp);
    Value vecNeutral = buildVectorChunkNeutral(rewriter, loc, cfg.chunkTy, op.getKind());

    auto forOp = rewriter.create<scf::ForOp>(
      loc, c0, cTotal, cChunk, vecNeutral, [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
        Value chunk = buildChunkTransferRead(b, bodyLoc, readOp, iv, cfg.chunkTy, inBoundsTrue);
        Value newAcc;
        if (isa<FloatType>(elemTy))
          newAcc = b.create<arith::AddFOp>(bodyLoc, chunk, iterArgs[0]);
        else
          newAcc = b.create<arith::AddIOp>(bodyLoc, chunk, iterArgs[0]);
        b.create<scf::YieldOp>(bodyLoc, newAcc);
      });

    rewriter.setInsertionPointAfter(forOp);
    // Seed read must use the same 1-D vector type as the store (e.g. vector<1xf32>), not a 0-D vector<f32>,
    // so permutation_map rank matches the vector type.
    Value seedRead =
      rewriter.create<vector::TransferReadOp>(loc, outVecTy, writeOp.getSource(), writeOp.getIndices(),
                                              writeOp.getPermutationMap(), pad, writeOp.getMask(), inBoundsWriteTrue);
    Value seedIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value seed = rewriter.create<vector::ExtractElementOp>(loc, seedRead, seedIdx);
    Value reduced = buildReduction(rewriter, loc, op.getKind(), cfg.chunkTy, forOp.getResult(0), seed);
    Value broadcasted = rewriter.create<vector::BroadcastOp>(loc, outVecTy, reduced);
    rewriter.create<vector::TransferWriteOp>(loc, broadcasted, writeOp.getSource(), writeOp.getIndices(),
                                             writeOp.getPermutationMapAttr(), writeOp.getMask(), inBoundsWriteTrue);

    rewriter.eraseOp(writeOp);
    rewriter.eraseOp(bcastOp);
    rewriter.eraseOp(op);
    if (readOp->use_empty()) rewriter.eraseOp(readOp);
    return success();
  }
};

struct NpuReductionToVector : public OpRewritePattern<npuvector::ReductionOp> {
  using OpRewritePattern<npuvector::ReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npuvector::ReductionOp op, PatternRewriter &rewriter) const override {
    Value v = op->getOperand(0);
    Type vecOpTy = v.getType();
    VectorType vecTy;
    if (auto npuTy = dyn_cast<npuvector::NPUVectorType>(vecOpTy))
      vecTy = toVectorType(npuTy);
    else if (auto vt = dyn_cast<VectorType>(vecOpTy))
      vecTy = vt;
    else
      return failure();
    if (!vecTy.hasStaticShape()) return failure();
    Value acc = op->getNumOperands() > 1 ? op->getOperand(1) : Value();
    Value newVal = buildReduction(rewriter, op.getLoc(), op.getKind(), vecTy, v, acc);
    rewriter.replaceOp(op, newVal);
    return success();
  }
};

static bool buildBroadcastDimMap(ArrayRef<int64_t> dimAttr, int64_t srcRank,
                                 SmallVector<int64_t, 4> &m) {
  if (dimAttr.empty()) {
    m.clear();
    for (int64_t i = 0; i < srcRank; ++i)
      m.push_back(i);
    return true;
  }
  if (static_cast<int64_t>(dimAttr.size()) != srcRank) return false;
  m.assign(dimAttr.begin(), dimAttr.end());
  return true;
}

static bool sortAxesByDimMap(const SmallVector<int64_t, 4> &m, int64_t srcRank,
                             SmallVector<int64_t, 4> &perm) {
  perm.resize(srcRank);
  for (int64_t i = 0; i < srcRank; ++i)
    perm[i] = i;
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return m[a] < m[b]; });
  for (int64_t t = 0; t + 1 < srcRank; ++t) {
    if (m[perm[t]] >= m[perm[t + 1]]) return false;
  }
  return true;
}

/// Reorder source axes so broadcast dimension indices follow destination axis order.
static Value applyBroadcastSourceTranspose(PatternRewriter &rewriter, Location loc, VectorType srcTy, Value srcVal,
                                           Type elemTy, ArrayRef<int64_t> perm, VectorType &laidOutTy) {
  const int64_t srcRank = srcTy.getRank();
  bool permIsIdentity = true;
  for (int64_t i = 0; i < srcRank; ++i) {
    if (perm[i] != i) {
      permIsIdentity = false;
      break;
    }
  }
  if (permIsIdentity) {
    laidOutTy = srcTy;
    return srcVal;
  }
  SmallVector<int64_t> transShape;
  transShape.reserve(srcRank);
  for (int64_t t = 0; t < srcRank; ++t)
    transShape.push_back(srcTy.getDimSize(perm[t]));
  laidOutTy = VectorType::get(transShape, elemTy);
  return rewriter
    .create<vector::TransposeOp>(loc, laidOutTy, srcVal, rewriter.getDenseI64ArrayAttr(perm))
    .getResult();
}

static bool fillBroadcastInsertShape(const SmallVector<int64_t, 4> &m,
                                     const SmallVector<int64_t, 4> &perm, int64_t srcRank, int64_t dstRank,
                                     VectorType laidOutTy, SmallVector<int64_t, 8> &iShape) {
  iShape.assign(dstRank, 1);
  for (int64_t j = 0; j < dstRank; ++j) {
    int64_t foundT = -1;
    for (int64_t t = 0; t < srcRank; ++t) {
      if (m[perm[t]] != j) continue;
      if (foundT >= 0) return false;
      foundT = t;
    }
    if (foundT >= 0) iShape[j] = laidOutTy.getDimSize(foundT);
  }
  return true;
}

/// Lower npuvector.broadcast to transpose (when axis map is out of order) + shape_cast insert 1s
/// on broadcast result axes + vector.broadcast, matching NumPy-style vector.broadcast rules.
static LogicalResult rewriteNpuBroadcastToVector(npuvector::BroadcastOp op, PatternRewriter &rewriter) {
  auto npuRes = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
  if (!npuRes) return failure();
  std::optional<VectorType> dstOpt = tryResolveStaticVectorType(npuRes, op.getMaxSizes());
  if (!dstOpt) return failure();
  VectorType dstTy = *dstOpt;

  Location loc = op.getLoc();
  Type elemTy = dstTy.getElementType();
  Value srcVal = op.getSource();
  Type srcType = srcVal.getType();

  if (!isa<VectorType>(srcType) && !isa<npuvector::NPUVectorType>(srcType)) {
    if (srcType != elemTy) return failure();
    Value out = rewriter.create<vector::BroadcastOp>(loc, dstTy, srcVal).getResult();
    rewriter.replaceOp(op, out);
    return success();
  }

  auto srcTy = dyn_cast<VectorType>(srcType);
  if (!srcTy || !srcTy.hasStaticShape()) return failure();

  const int64_t srcRank = srcTy.getRank();
  const int64_t dstRank = dstTy.getRank();
  if (srcRank > dstRank) return failure();

  SmallVector<int64_t, 4> m;
  if (!buildBroadcastDimMap(op.getDimensionAttr().asArrayRef(), srcRank, m)) return failure();

  SmallVector<int64_t, 4> perm;
  if (!sortAxesByDimMap(m, srcRank, perm)) return failure();

  VectorType laidOutTy;
  Value laidOutVal = applyBroadcastSourceTranspose(rewriter, loc, srcTy, srcVal, elemTy, perm, laidOutTy);

  SmallVector<int64_t, 8> iShape;
  if (!fillBroadcastInsertShape(m, perm, srcRank, dstRank, laidOutTy, iShape)) return failure();

  VectorType interTy = VectorType::get(iShape, elemTy);
  Value expanded = laidOutVal;
  if (cast<VectorType>(laidOutVal.getType()) != interTy) {
    expanded = rewriter.create<vector::ShapeCastOp>(loc, interTy, laidOutVal).getResult();
  }
  Value out = rewriter.create<vector::BroadcastOp>(loc, dstTy, expanded).getResult();
  rewriter.replaceOp(op, out);
  return success();
}

struct NpuBroadcastToVector : public OpRewritePattern<npuvector::BroadcastOp> {
  using OpRewritePattern<npuvector::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, PatternRewriter &rewriter) const override {
    return rewriteNpuBroadcastToVector(op, rewriter);
  }
};

struct NpuTransposeToVector : public OpRewritePattern<npuvector::TransposeOp> {
  using OpRewritePattern<npuvector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(npuvector::TransposeOp op, PatternRewriter &rewriter) const override {
    auto outNpu = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!outNpu) return failure();
    Value inVal = op->getOperand(0);
    Type inTy = inVal.getType();
    auto inStatic = dyn_cast<VectorType>(inTy);
    if (!inStatic || !inStatic.hasStaticShape()) return failure();
    ArrayRef<int64_t> perm = op.getPermutation();
    SmallVector<int64_t, 4> permutedShape;
    permutedShape.reserve(perm.size());
    std::transform(perm.begin(), perm.end(), std::back_inserter(permutedShape),
                   [&](int64_t p) { return inStatic.getDimSize(p); });
    VectorType outVec = VectorType::get(permutedShape, outNpu.getElementType());
    if (inVal.getType() != inStatic) return failure();
    auto newOp = rewriter.create<vector::TransposeOp>(op.getLoc(), outVec, inVal, op.getPermutation());
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// Lower elementwise npuvector cast/ compare / select to arith on vector types.
// Without these, arith ops can end up with vector-typed results but npuvector-typed
// operands; ArithWithNpuTypesToVector only rewrites results and would match forever.

// Single-operand npuvector cast with matching result shape; lowers to arith (loc, outVec, in).
// Operand may already be a plain VectorType (e.g. after npuvector.transfer_read lowers) with a
// different element type than outVec (extf, truncf, int/fp casts); only shapes must match.
// extf / truncf use non-default arith builders (FastMathFlagsAttr / minimal truncf).
template <typename NpuOp, typename ArithOp>
struct NpuUnaryInOutToArith : public OpRewritePattern<NpuOp> {
  using OpRewritePattern<NpuOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuOp op, PatternRewriter &rewriter) const override {
    auto outNpu = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!outNpu) return failure();
    VectorType outVec = inferStaticResultVecFromOperands(outNpu, TypeRange(op->getOperands()));
    if (!outVec.hasStaticShape()) return failure();
    Value in = op->getOperand(0);
    auto inVecTy = dyn_cast<VectorType>(in.getType());
    if (!inVecTy || !inVecTy.hasStaticShape()) return failure();
    if (inVecTy.getShape() != outVec.getShape()) return failure();

    Value repl;
    if constexpr (std::is_same_v<NpuOp, npuvector::ExtFOp>) {
      auto fmfAttr = arith::FastMathFlagsAttr::get(op.getContext(), op.getFastmath());
      repl = rewriter.create<arith::ExtFOp>(op.getLoc(), outVec, in, fmfAttr).getResult();
    } else if constexpr (std::is_same_v<NpuOp, npuvector::TruncFOp>) {
      repl = rewriter.create<arith::TruncFOp>(op.getLoc(), outVec, in).getResult();
    } else {
      repl = rewriter.create<ArithOp>(op.getLoc(), outVec, in).getResult();
    }
    rewriter.replaceOp(op, repl);
    return success();
  }
};

template <typename NpuOp, bool IsFloatCmp>
struct NpuCmpToArith : public OpRewritePattern<NpuOp> {
  using OpRewritePattern<NpuOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuOp op, PatternRewriter &rewriter) const override {
    auto outNpu = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!outNpu) return failure();
    Type lt = op->getOperand(0).getType(), rt = op->getOperand(1).getType();
    auto lhsNpu = dyn_cast<npuvector::NPUVectorType>(lt);
    auto rhsNpu = dyn_cast<npuvector::NPUVectorType>(rt);
    npuvector::NPUVectorType keyTy = lhsNpu ? lhsNpu : rhsNpu;
    VectorType operandVec;
    if (keyTy) {
      operandVec = inferStaticResultVecFromOperands(keyTy, TypeRange(op->getOperands()));
    } else {
      auto lv = dyn_cast<VectorType>(lt);
      auto rv = dyn_cast<VectorType>(rt);
      if (lv && lv.hasStaticShape())
        operandVec = lv;
      else if (rv && rv.hasStaticShape())
        operandVec = rv;
      else
        return failure();
    }
    if (!operandVec.hasStaticShape()) return failure();
    VectorType outVec = VectorType::get(operandVec.getShape(), rewriter.getI1Type());
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    if (lhs.getType() != operandVec || rhs.getType() != operandVec) return failure();
    if constexpr (IsFloatCmp) {
      auto newOp = rewriter.create<arith::CmpFOp>(op.getLoc(), outVec, op.getPredicate(), lhs, rhs, op.getFastmath());
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      auto newOp = rewriter.create<arith::CmpIOp>(op.getLoc(), outVec, op.getPredicate(), lhs, rhs);
      rewriter.replaceOp(op, newOp.getResult());
    }
    return success();
  }
};

template <typename NpuOp, typename ArithOp>
struct NpuSelectToArithT : public OpRewritePattern<NpuOp> {
  using OpRewritePattern<NpuOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NpuOp op, PatternRewriter &rewriter) const override {
    auto outNpu = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!outNpu) return failure();
    VectorType outVec = inferStaticResultVecFromOperands(outNpu, TypeRange(op->getOperands()));
    if (!outVec.hasStaticShape()) return failure();
    Value c = op->getOperand(0), t = op->getOperand(1), f = op->getOperand(2);
    VectorType condTy = VectorType::get(outVec.getShape(), rewriter.getI1Type());
    if (c.getType() != condTy || t.getType() != outVec || f.getType() != outVec) return failure();
    auto newOp = rewriter.create<ArithOp>(op.getLoc(), outVec, c, t, f);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// After npuvector constants become vector<...> inits, region iter_args can still be !npuvector<...>,
/// which breaks scf.for verification. Align iter_arg and for op result types with init value types.
struct ScfForUnifyIterArgsWithInitVectorTypes : public OpRewritePattern<scf::ForOp> {
  explicit ScfForUnifyIterArgsWithInitVectorTypes(MLIRContext *ctx, PatternBenefit b = PatternBenefit(10))
      : OpRewritePattern<scf::ForOp>(ctx, b) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    auto inits = forOp.getInits();
    auto iterArgs = forOp.getRegionIterArgs();
    auto results = forOp.getResults();
    if (inits.size() != iterArgs.size() || inits.size() != results.size()) return failure();

    bool changed = false;
    for (auto [init, iterArg, res] : llvm::zip(inits, iterArgs, results)) {
      if (!isa<npuvector::NPUVectorType>(iterArg.getType())) continue;
      auto initVec = dyn_cast<VectorType>(init.getType());
      if (!initVec || !initVec.hasStaticShape()) continue;
      auto nvt = cast<npuvector::NPUVectorType>(iterArg.getType());
      if (nvt.getElementType() != initVec.getElementType() || nvt.getRank() != initVec.getRank()) continue;
      if (iterArg.getType() == init.getType()) continue;
      iterArg.setType(init.getType());
      res.setType(init.getType());
      changed = true;
    }
    return changed ? success() : failure();
  }
};

struct ArithConstantNpuToVector : public OpRewritePattern<arith::ConstantOp> {
  using OpRewritePattern<arith::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, PatternRewriter &rewriter) const override {
    auto npuTy = dyn_cast<npuvector::NPUVectorType>(op.getType());
    if (!npuTy) return failure();
    auto vecTyOpt = tryResolveStaticVectorType(npuTy, ValueRange{});
    if (!vecTyOpt) return failure();
    auto vecTy = *vecTyOpt;
    auto val = op.getValue();
    if (auto d = dyn_cast<DenseElementsAttr>(val)) {
      if (d.getType() != npuTy) return failure();
      auto reshaped = d.reshape(vecTy);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vecTy, reshaped);
    } else {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vecTy, val);
    }
    return success();
  }
};

/// Reduce a 1-D vector larger than the hardware chunk (64xf32 / 128xf16): accumulate chunk vectors
/// in scf.for, then apply vector.multi_reduction once outside the loop.
struct ChunkLargeVectorReduction : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op, PatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<VectorType>(op.getSource().getType());
    if (!srcTy) return failure();
    auto cfgOpt = get1DVectorChunkConfig(srcTy);
    if (!cfgOpt) return failure();

    if (op.getReductionDims().size() != 1) return failure();
    auto redDims = op.getReductionDims().getAsValueRange<IntegerAttr>();
    if (*redDims.begin() != 0) return failure();
    if (srcTy.getRank() != 1) return failure();
    if (op.getDest().getType() != srcTy.getElementType()) return failure();

    auto readOp = getDefiningVectorTransferRead(op.getSource());
    if (!readOp || readOp.getVectorType() != srcTy) return failure();

    const VectorChunkConfig &cfg = *cfgOpt;
    Location loc = op.getLoc();
    Type elemTy = srcTy.getElementType();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, cfg.totalSize);
    Value cChunk = rewriter.create<arith::ConstantIndexOp>(loc, cfg.chunkSize);
    ArrayAttr inBoundsTrue = getAllTrueInBoundsAttr(rewriter, cfg.chunkTy.getRank());

    rewriter.setInsertionPoint(readOp);
    Value vecNeutral = buildVectorChunkNeutral(rewriter, loc, cfg.chunkTy, op.getKind());
    vector::CombiningKind kind = op.getKind();

    auto forOp = rewriter.create<scf::ForOp>(
      loc, c0, cTotal, cChunk, vecNeutral, [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange iterArgs) {
        Value chunk = buildChunkTransferRead(b, bodyLoc, readOp, iv, cfg.chunkTy, inBoundsTrue);
        Value newAcc;
        if (isa<FloatType>(elemTy))
          newAcc = b.create<arith::AddFOp>(bodyLoc, chunk, iterArgs[0]);
        else
          newAcc = b.create<arith::AddIOp>(bodyLoc, chunk, iterArgs[0]);
        b.create<scf::YieldOp>(bodyLoc, newAcc);
      });

    rewriter.setInsertionPointAfter(forOp);
    Value scalarAcc = op.getAcc() ? op.getAcc() : getCombiningNeutral(rewriter, loc, kind, elemTy);
    Value reduced = buildReduction(rewriter, loc, kind, cfg.chunkTy, forOp.getResult(0), scalarAcc);
    rewriter.replaceOp(op, reduced);
    if (readOp->use_empty()) rewriter.eraseOp(readOp);
    return success();
  }
};

struct ChunkLargeVectorTransferWrite : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op, PatternRewriter &rewriter) const override {
    auto vecTy = dyn_cast<VectorType>(op.getVector().getType());
    if (!vecTy) return failure();
    auto cfgOpt = get1DVectorChunkConfig(vecTy);
    if (!cfgOpt) return failure();
    const VectorChunkConfig &cfg = *cfgOpt;

    Location loc = op.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cTotal = rewriter.create<arith::ConstantIndexOp>(loc, cfg.totalSize);
    Value cChunk = rewriter.create<arith::ConstantIndexOp>(loc, cfg.chunkSize);

    Value vector = op.getVector();
    std::optional<Value> splatChunk = getSplatChunkConstant(rewriter, loc, vector, cfg);
    if (!splatChunk) return failure();

    rewriter.setInsertionPoint(op);
    auto forOp = rewriter.create<scf::ForOp>(loc, c0, cTotal, cChunk, std::nullopt,
                                             [&](OpBuilder &b, Location bodyLoc, Value iv, ValueRange) {
                                               buildChunkTransferWrite(b, bodyLoc, op, iv, *splatChunk);
                                               b.create<scf::YieldOp>(bodyLoc);
                                             });

    rewriter.setInsertionPointAfter(forOp);
    if (op->getNumResults() != 0) {
      rewriter.replaceOp(op, forOp.getResults());
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

struct ArithWithNpuTypesToVector : public RewritePattern {
  explicit ArithWithNpuTypesToVector(MLIRContext *ctx)
      : RewritePattern(Pattern::MatchAnyOpTypeTag{}, 1, ctx) {}

  bool hasNpuVector(TypeRange types) const {
    return llvm::any_of(types, [](Type t) { return isa<npuvector::NPUVectorType>(t); });
  }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if ((!isa<arith::ArithDialect, math::MathDialect>(op->getDialect())) || isa<arith::ConstantOp>(op))
      return failure();
    if (op->getNumRegions() != 0) return failure();
    // Only rewrite arith op *results* still using !npuvector. Matching on operands alone
    // causes infinite apply: after one step results become vector but operands can stay
    // npuvector until npuvector.* ops are lowered, and the pattern keeps re-creating the
    // same arith op.
    if (!hasNpuVector(op->getResultTypes())) return failure();

    SmallVector<Type, 4> newResults;
    newResults.reserve(op->getNumResults());
    for (Type t : op->getResultTypes()) {
      auto nvt = dyn_cast<npuvector::NPUVectorType>(t);
      if (!nvt) {
        newResults.push_back(t);
        continue;
      }
      VectorType inf = inferStaticResultVecFromOperands(nvt, TypeRange(op->getOperandTypes()));
      if (!inf.hasStaticShape()) return failure();
      newResults.push_back(inf);
    }

    OperationState st(op->getLoc(), op->getName().getIdentifier());
    st.addOperands(op->getOperands());
    st.addTypes(newResults);
    st.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class NPUVectorToVectorPass : public impl::NPUVectorToVectorBase<NPUVectorToVectorPass> {
 public:
  NPUVectorToVectorPass() = default;
  NPUVectorToVectorPass(const NPUVectorToVectorPass &) = default;

  StringRef getArgument() const override { return "npuvector-to-vector"; }

  StringRef getDescription() const override { return "Lower NPUVector ops to MLIR vector/arith on community types"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, npuvector::NPUVectorDialect, vector::VectorDialect,
                    memref::MemRefDialect, func::FuncDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult walkResult = module.walk([&](func::FuncOp func) -> WalkResult {
      if (!func->hasAttr(kVectorFunctionAttr)) return WalkResult::advance();
      RewritePatternSet ps(func.getContext());
      ps.add<
        ScfForUnifyIterArgsWithInitVectorTypes, NpuTransferReadToVector, NpuTransferWriteToVector, NpuReductionToVector,
        NpuBroadcastToVector, NpuTransposeToVector, NpuUnaryInOutToArith<npuvector::ExtUIOp, arith::ExtUIOp>,
        NpuUnaryInOutToArith<npuvector::ExtSIOp, arith::ExtSIOp>,
        NpuUnaryInOutToArith<npuvector::TruncIOp, arith::TruncIOp>,
        NpuUnaryInOutToArith<npuvector::ExtFOp, arith::ExtFOp>,
        NpuUnaryInOutToArith<npuvector::TruncFOp, arith::TruncFOp>,
        NpuUnaryInOutToArith<npuvector::SIToFPOp, arith::SIToFPOp>,
        NpuUnaryInOutToArith<npuvector::UIToFPOp, arith::UIToFPOp>,
        NpuUnaryInOutToArith<npuvector::FPToSIOp, arith::FPToSIOp>,
        NpuUnaryInOutToArith<npuvector::FPToUIOp, arith::FPToUIOp>,
        NpuUnaryInOutToArith<npuvector::BitcastOp, arith::BitcastOp>,
        NpuUnaryInOutToArith<npuvector::IndexCastOp, arith::IndexCastOp>,
        NpuUnaryInOutToArith<npuvector::IndexCastUIOp, arith::IndexCastUIOp>, NpuCmpToArith<npuvector::CmpIOp, false>,
        NpuCmpToArith<npuvector::CmpFOp, true>, NpuSelectToArithT<npuvector::SelectOp, arith::SelectOp>,
        ArithConstantNpuToVector, ArithWithNpuTypesToVector>(func.getContext());
      if (failed(applyPatternsAndFoldGreedily(func, std::move(ps)))) return WalkResult::interrupt();

      RewritePatternSet chunkPs(func.getContext());
      chunkPs.add<FuseChunkedMemrefElementwise, FuseChunkedReductionMemrefStore, ChunkLargeVectorReduction,
                  ChunkLargeVectorTransferWrite>(func.getContext());
      if (failed(applyPatternsAndFoldGreedily(func, std::move(chunkPs)))) return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::createNPUVectorToVectorPass() {
  return std::make_unique<NPUVectorToVectorPass>();
}
