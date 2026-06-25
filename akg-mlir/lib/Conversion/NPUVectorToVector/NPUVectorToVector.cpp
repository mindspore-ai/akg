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
#include <cstdint>
#include <optional>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_NPUVECTORTOVECTOR
#include "akg/Conversion/Passes.h.inc"

namespace {

namespace npuv = mlir::npuvector;

struct TileDesc {
  npuv::NPUVectorType type;
  ValueRange dynSizes;
  Value source;
};

struct ReductionDesc {
  npuv::ReductionOp redOp;
  Type scalarElem;
  Value idScalar;
  Value c0;
};

// ===========================================================================
// Register-width helpers (Step 3)
// ===========================================================================

// Get the register vector width (in bytes) of the device described by `func`'s
// `arch` attribute. Mirrors `getVectorWidth` in OutlineVectorFunction.cpp.
static std::optional<int64_t> getRegisterWidthBytes(func::FuncOp funcOp) {
  if (!funcOp) {
    return std::nullopt;
  }
  if (auto archAttr = funcOp->getAttrOfType<StringAttr>("arch")) {
    uint32_t width = akg::NpuInfo::getInstance(archAttr.getValue().str()).getRegVectorLength();
    if (width > 0) {
      return static_cast<int64_t>(width);
    }
  }
  return std::nullopt;
}

// A vf kernel is private and does not carry `arch` itself; the attribute lives
// on the (non-vf) parent function that calls it. Walk the callers to find a
// function that exposes `arch`.
static func::FuncOp findArchFunc(func::FuncOp vfFunc) {
  if (vfFunc->hasAttr("arch")) {
    return vfFunc;
  }
  auto module = vfFunc->getParentOfType<ModuleOp>();
  if (!module) {
    return {};
  }
  func::FuncOp found;
  module.walk([&found, &vfFunc](func::CallOp call) {
    if (found) {
      return;
    }
    if (call.getCallee() != vfFunc.getName()) {
      return;
    }
    auto parent = call->getParentOfType<func::FuncOp>();
    if (parent && parent->hasAttr("arch")) {
      found = parent;
    }
  });
  return found;
}

// Byte width of an element type, or 0 for sub-byte / non int-or-float types
// (e.g. `i1` masks) that must not drive the register lane count.
static unsigned elemByteWidth(Type elemType) {
  if (!elemType || !elemType.isIntOrFloat()) {
    return 0;
  }
  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  if (bitWidth < 8) {
    return 0;
  }
  return bitWidth / 8;
}

// Compute one lane count for the whole vf kernel so that every dynamic
// `!npuvector<?x...>` extent resolves to the *same* static size. Because the
// whole kernel shares a single lane count (one loop / index space), it must be
// sized for the *widest* (sub-byte excluded) data type so that NO register
// exceeds the physical width: `regWidthBytes / maxElemBytes`. Using the
// narrowest type instead would overflow the wider registers, e.g. a kernel
// mixing i8 (1B) and f32 (4B) on a 256B register would pick 256 lanes and blow
// the f32 register up to 256*4 = 1024B (4x over budget -> stack overflow).
// With a 256B register this yields the Step 3 expectation:
//   pure f32 kernel -> 64 lanes;  pure f16 kernel -> 128 lanes.
//   mixed f32+i8   -> 64 lanes (f32=256B, i8=64B, both within budget).
static int64_t computeLaneCount(func::FuncOp vfFunc, int64_t regWidthBytes) {
  unsigned maxBytes = 0;
  auto consider = [&maxBytes](Type t) {
    if (auto vt = llvm::dyn_cast<npuv::NPUVectorType>(t)) {
      unsigned b = elemByteWidth(vt.getElementType());
      if (b > maxBytes) {
        maxBytes = b;
      }
    }
  };
  vfFunc.walk([&consider](Operation *op) {
    for (Type t : op->getOperandTypes()) {
      consider(t);
    }
    for (Type t : op->getResultTypes()) {
      consider(t);
    }
  });
  if (maxBytes == 0) {
    maxBytes = 1;
  }
  int64_t lanes = regWidthBytes / static_cast<int64_t>(maxBytes);
  return lanes > 0 ? lanes : 1;
}

static bool isNpuVectorType(Type t) { return t && llvm::isa<npuv::NPUVectorType>(t); }

static bool touchesNpuVector(Operation *op) {
  auto isVec = [](Value v) { return isNpuVectorType(v.getType()); };
  return llvm::any_of(op->getOperands(), isVec) || llvm::any_of(op->getResults(), isVec);
}

// A "unit" npuvector is fully static and holds exactly one element (e.g.
// `!npuvector<1xf32>`). Reads producing such values are scalar / broadcast
// source loads, NOT the kernel's iteration space: the tiler must neither anchor
// its loop on them nor index them with the tile induction variables.
static bool isUnitNPUVector(Type t) {
  auto vt = llvm::dyn_cast_or_null<npuv::NPUVectorType>(t);
  if (!vt) {
    return false;
  }
  int64_t total = 1;
  for (int64_t d : vt.getShape()) {
    if (ShapedType::isDynamic(d)) {
      return false;
    }
    total *= d;
  }
  return total == 1;
}

// True when `v` is consumed *exclusively* by npuvector broadcast ops. Only such
// a unit read is a pure scalar / broadcast source that may stay vector<1>: the
// following broadcasts stretch its lone element across all lanes. If `v` has any
// other (e.g. arith element-wise) consumer it must instead be tiled to the
// register width, otherwise that consumer would mix a vector<1> operand with the
// laneCount-wide operands and fail verification.
static bool onlyFeedsBroadcast(Value v) {
  return !v.use_empty() && llvm::all_of(v.getUsers(), [](Operation *u) { return isa<npuv::BroadcastOp>(u); });
}

// Convert an `!npuvector` type to a community `vector` type, resolving every
// dynamic extent to `laneCount` (static extents are preserved). Used by the
// fallback (non-tiled) lowering path.
static VectorType convertToVectorType(npuv::NPUVectorType vt, int64_t laneCount) {
  SmallVector<int64_t> shape(vt.getShape().begin(), vt.getShape().end());
  for (int64_t &d : shape) {
    if (ShapedType::isDynamic(d)) {
      d = laneCount;
    }
  }
  if (shape.empty()) {
    shape.push_back(laneCount);
  }
  return VectorType::get(shape, vt.getElementType());
}

// ===========================================================================
// Reduction identity helper
// ===========================================================================

static TypedAttr getCombiningIdentityAttr(OpBuilder &b, vector::CombiningKind kind, Type elemType) {
  using CK = vector::CombiningKind;
  if (auto ft = llvm::dyn_cast<FloatType>(elemType)) {
    const llvm::fltSemantics &sem = ft.getFloatSemantics();
    switch (kind) {
      case CK::MUL:
        return b.getFloatAttr(elemType, 1.0);
      case CK::MINNUMF:
      case CK::MINIMUMF:
        return FloatAttr::get(elemType, APFloat::getInf(sem, /* Negative= */ false));
      case CK::MAXNUMF:
      case CK::MAXIMUMF:
        return FloatAttr::get(elemType, APFloat::getInf(sem, /* Negative= */ true));
      default:
        return b.getFloatAttr(elemType, 0.0);
    }
  }
  unsigned width = elemType.isIndex() ? 64u : elemType.getIntOrFloatBitWidth();
  switch (kind) {
    case CK::MUL:
      return b.getIntegerAttr(elemType, 1);
    case CK::AND:
    case CK::MINUI:
      return IntegerAttr::get(elemType, APInt::getAllOnes(width));
    case CK::MAXSI:
      return IntegerAttr::get(elemType, APInt::getSignedMinValue(width));
    case CK::MINSI:
      return IntegerAttr::get(elemType, APInt::getSignedMaxValue(width));
    default:
      // ADD / OR / XOR / MAXUI all have a zero identity.
      return b.getIntegerAttr(elemType, 0);
  }
}

static Value buildIdentityAccumulator(OpBuilder &b, Location loc, vector::CombiningKind kind, Type destType) {
  Type elemType = getElementTypeOrSelf(destType);
  TypedAttr scalarAttr = getCombiningIdentityAttr(b, kind, elemType);
  // For a vector accumulator emit a splat dense constant directly rather than a
  // `vector.broadcast` of a scalar: the broadcast op is lowered to an alloc +
  // VBrc downstream and breaks reduction legalization (the `iter_args` init must
  // stay a plain constant, matching the working store-after-reduction shape).
  if (auto vt = llvm::dyn_cast<VectorType>(destType)) {
    return b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vt, scalarAttr));
  }
  return b.create<arith::ConstantOp>(loc, scalarAttr);
}

// ===========================================================================
// Fallback lowering patterns (used for static / reduction / transpose kernels)
// ===========================================================================

struct TransferReadLowering : public OpConversionPattern<npuv::TransferReadOp> {
  using OpConversionPattern<npuv::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::TransferReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getVector().getType());
    auto vecType = llvm::dyn_cast_or_null<VectorType>(converted);
    if (!vecType) {
      return rewriter.notifyMatchFailure(op, "result type is not convertible to vector");
    }
    auto permMap = rewriter.getMultiDimIdentityMap(vecType.getRank());
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(vecType.getRank(), true));
    auto newRead = rewriter.create<vector::TransferReadOp>(op.getLoc(), vecType, adaptor.getSource(),
                                                           adaptor.getIndices(), permMap, adaptor.getPadding(),
                                                           /* mask= */ Value(), inBounds);
    rewriter.replaceOp(op, newRead.getResult());
    return success();
  }
};

struct TransferWriteLowering : public OpConversionPattern<npuv::TransferWriteOp> {
  using OpConversionPattern<npuv::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::TransferWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto vecType = llvm::dyn_cast<VectorType>(adaptor.getVector().getType());
    if (!vecType) {
      return rewriter.notifyMatchFailure(op, "stored value is not a vector");
    }
    auto permMap = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(vecType.getRank()));
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(vecType.getRank(), true));
    rewriter.create<vector::TransferWriteOp>(op.getLoc(), adaptor.getVector(), adaptor.getSource(),
                                             adaptor.getIndices(), permMap, inBounds);
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReductionLowering : public OpConversionPattern<npuv::ReductionOp> {
  using OpConversionPattern<npuv::ReductionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::ReductionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcType = llvm::dyn_cast<VectorType>(adaptor.getVector().getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(op, "reduction source is not a vector");
    }
    Type destType = getTypeConverter()->convertType(op.getDest().getType());
    if (!destType) {
      return rewriter.notifyMatchFailure(op, "reduction result type not convertible");
    }

    int64_t rank = srcType.getRank();
    SmallVector<bool> reductionMask(rank, false);
    if (auto dims = op.getReductionDims(); dims && !dims->empty()) {
      for (int64_t d : *dims) {
        if (d >= 0 && d < rank) {
          reductionMask[d] = true;
        }
      }
    } else {
      reductionMask.assign(rank, true);
    }

    Value acc = adaptor.getAcc();
    if (!acc) {
      acc = buildIdentityAccumulator(rewriter, op.getLoc(), op.getKind(), destType);
    }

    auto reduce =
      rewriter.create<vector::MultiDimReductionOp>(op.getLoc(), adaptor.getVector(), acc, reductionMask, op.getKind());
    rewriter.replaceOp(op, reduce.getResult());
    return success();
  }
};

struct BroadcastLowering : public OpConversionPattern<npuv::BroadcastOp> {
  using OpConversionPattern<npuv::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    auto vecType = llvm::dyn_cast_or_null<VectorType>(converted);
    if (!vecType) {
      return rewriter.notifyMatchFailure(op, "broadcast result not convertible to vector");
    }
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, vecType, adaptor.getSource());
    return success();
  }
};

struct TransposeLowering : public OpConversionPattern<npuv::TransposeOp> {
  using OpConversionPattern<npuv::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!llvm::isa<VectorType>(adaptor.getVector().getType())) {
      return rewriter.notifyMatchFailure(op, "transpose operand not a vector");
    }
    SmallVector<int64_t> perm(op.getPermutation().begin(), op.getPermutation().end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, adaptor.getVector(), perm);
    return success();
  }
};

template <typename SrcOp, typename DstOp>
struct UnaryCastLowering : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using OpAdaptor = typename SrcOp::Adaptor;

  LogicalResult matchAndRewrite(SrcOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op.getOut().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "cast result type not convertible");
    }
    rewriter.replaceOpWithNewOp<DstOp>(op, resultType, adaptor.getIn());
    return success();
  }
};

struct CmpILowering : public OpConversionPattern<npuv::CmpIOp> {
  using OpConversionPattern<npuv::CmpIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::CmpIOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, op.getPredicate(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct CmpFLowering : public OpConversionPattern<npuv::CmpFOp> {
  using OpConversionPattern<npuv::CmpFOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::CmpFOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, op.getPredicate(), adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

struct SelectLowering : public OpConversionPattern<npuv::SelectOp> {
  using OpConversionPattern<npuv::SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuv::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(), adaptor.getTrueValue(),
                                                 adaptor.getFalseValue());
    return success();
  }
};

struct ElementwiseRetypeLowering : public ConversionPattern {
  ElementwiseRetypeLowering(const TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /* benefit= */ 0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    Dialect *dialect = op->getDialect();
    if (!llvm::isa_and_nonnull<arith::ArithDialect, math::MathDialect>(dialect)) {
      return failure();
    }
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      return failure();
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) {
      return failure();
    }

    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(op)) {
      auto vecType = llvm::dyn_cast<VectorType>(resultTypes.front());
      if (!vecType) {
        return failure();
      }
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense) {
        return failure();
      }
      DenseElementsAttr newDense =
        dense.isSplat() ? DenseElementsAttr::get(vecType, dense.getSplatValue<Attribute>()) : dense.reshape(vecType);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(cst, vecType, newDense);
      return success();
    }

    OperationState state(op->getLoc(), op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// ===========================================================================
// Pass
// ===========================================================================

class NPUVectorToVector : public impl::NPUVectorToVectorBase<NPUVectorToVector> {
 public:
  NPUVectorToVector() = default;
  NPUVectorToVector(const NPUVectorToVector &) = default;
  NPUVectorToVector &operator=(const NPUVectorToVector &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, npuv::NPUVectorDialect, vector::VectorDialect,
                    memref::MemRefDialect, func::FuncDialect, scf::SCFDialect>();
  }

  // -------------------------------------------------------------------------
  // Tiling path: 1-D dynamic elementwise kernels are split into a register
  // width loop, e.g. for f32 (256B register, 64 lanes):
  //   scf.for %iv = 0 to %len step 64 {
  //     %n    = minsi (%len - %iv), 64
  //     %mask = vector.create_mask %n : vector<64xi1>
  //     %a    = vector.transfer_read %arg0[%iv], %pad, %mask : ... vector<64xf32>
  //     ...
  //     vector.transfer_write %res, %arg3[%iv], %mask
  //   }
  // -------------------------------------------------------------------------

  // A kernel is tileable when its compute chain is purely element-wise (any
  // rank, static or dynamic; no reduction / transpose / non-splat constant)
  // and has a transfer_read to anchor the tile shape. Reduction / transpose
  // need a different loop structure and go through the fallback path.
  static bool isTileable(func::FuncOp vfFunc) {
    if (vfFunc.getBody().empty()) {
      return false;
    }
    bool ok = true;
    bool hasAnchor = false;
    vfFunc.walk([&ok, &hasAnchor](Operation *op) {
      if (isa<npuv::ReductionOp, npuv::TransposeOp>(op)) {
        ok = false;
      }
      if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        if (isNpuVectorType(cst.getType())) {
          auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
          if (!dense || !dense.isSplat()) {
            ok = false;
          }
        }
      }
      if (auto rd = dyn_cast<npuv::TransferReadOp>(op)) {
        auto vt = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType());
        hasAnchor = true;
        // Every dynamic extent needs a matching dynamicSizes operand to bound
        // the corresponding loop.
        if (vt.getNumDynamicDims() > rd.getDynamicSizes().size()) {
          ok = false;
        }
      }
      if (auto wr = dyn_cast<npuv::TransferWriteOp>(op)) {
        if (isNpuVectorType(wr.getVector().getType())) {
          hasAnchor = true;
        }
      }
    });
    return ok && hasAnchor;
  }

  // A kernel goes through the tiled reduction path when it contains exactly one
  // full reduction-to-scalar (no transpose / partial reduction), the producers
  // feeding it are tileable element-wise ops, and the ops consuming the scalar
  // result do not touch npuvector (so they can stay after the loop unchanged).
  // Walk the kernel collecting the single reduction op while validating the
  // structural preconditions shared by every tileable reduction: no transpose,
  // npuvector constants must be splats, and dynamic reads must be fully bounded.
  // Returns true only when exactly one reduction and at least one read are found
  // and no precondition is violated; `redOp` is set to that reduction.
  static bool scanReductionCandidate(func::FuncOp vfFunc, npuv::ReductionOp &redOp) {
    int redCount = 0;
    bool ok = true;
    bool hasRead = false;
    vfFunc.walk([&ok, &redOp, &redCount, &hasRead](Operation *op) {
      if (isa<npuv::TransposeOp>(op)) {
        ok = false;
      }
      if (auto r = dyn_cast<npuv::ReductionOp>(op)) {
        redOp = r;
        ++redCount;
      }
      if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
        if (isNpuVectorType(cst.getType())) {
          auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
          if (!dense || !dense.isSplat()) {
            ok = false;
          }
        }
      }
      if (auto rd = dyn_cast<npuv::TransferReadOp>(op)) {
        auto vt = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType());
        hasRead = true;
        if (vt.getNumDynamicDims() > rd.getDynamicSizes().size()) {
          ok = false;
        }
      }
    });
    return redCount == 1 && ok && hasRead;
  }

  static bool isReductionTileable(func::FuncOp vfFunc) {
    if (vfFunc.getBody().empty()) {
      return false;
    }
    npuv::ReductionOp redOp;
    if (!scanReductionCandidate(vfFunc, redOp)) {
      return false;
    }
    // Only full reduction (all dims -> scalar) is tiled here; partial reduction
    // keeps a lower-rank vector result and stays on the fallback path.
    if (auto dims = redOp.getReductionDims(); dims && !dims->empty()) {
      return false;
    }
    if (isNpuVectorType(redOp.getDest().getType())) {
      return false;
    }
    // The scalar result must be consumed by non-npuvector ops only.
    Block &entry = vfFunc.getBody().front();
    bool afterReduction = false;
    for (Operation &op : entry.without_terminator()) {
      if (&op == redOp.getOperation()) {
        afterReduction = true;
        continue;
      }
      if (afterReduction && touchesNpuVector(&op)) {
        return false;
      }
    }
    return true;
  }

  // Rebuild one npuvector / arith-on-npuvector op as a `laneCount`-wide vector
  // op inside the tiling loop nest. `indices` is the running offset per memref
  // dimension (one induction var per dimension, innermost last), `mask` masks
  // the active lanes of the innermost (register) dimension.
  LogicalResult tileComputeOp(OpBuilder &b, Operation *op, IRMapping &map, ValueRange indices, Value mask,
                              int64_t laneCount) const {
    Location loc = op->getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    auto vecTypeOf = [laneCount](Type elemType) { return VectorType::get({laneCount}, elemType); };
    // The register vector is always 1-D (vector<laneCount>); the permutation
    // map projects the n-D memref index space onto its innermost dimension.
    // A memref source may have FEWER dims than the anchor iteration space when
    // it is a broadcast operand that aligns with the innermost dims (e.g. a 1-D
    // bias added to a 2-D tensor). Such an op must be indexed with exactly its
    // own source rank, using the trailing induction variables, so the emitted
    // vector.transfer_{read,write} carries the right number of indices.
    auto sourceRank = [&indices](Value src) -> int64_t {
      auto mt = llvm::dyn_cast<MemRefType>(src.getType());
      return mt ? mt.getRank() : static_cast<int64_t>(indices.size());
    };
    auto trailingIndices = [&indices](int64_t srcRank) {
      SmallVector<Value> result(indices.begin(), indices.end());
      auto n = static_cast<int64_t>(result.size());
      if (srcRank < n) {
        result.erase(result.begin(), result.begin() + (n - srcRank));
      }
      return result;
    };
    auto permMapForRank = [&b](int64_t srcRank) {
      return AffineMap::getMinorIdentityMap(static_cast<unsigned>(srcRank), 1, b.getContext());
    };
    auto inBounds = b.getBoolArrayAttr({true});

    if (auto rd = dyn_cast<npuv::TransferReadOp>(op)) {
      auto nvt = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType());
      auto elemType = nvt.getElementType();
      // A unit (single-element) read whose ONLY consumers are broadcasts is a
      // scalar load. Read it at its OWN fixed indices and keep its static shape
      // so the following broadcast can stretch the lone element across all
      // `laneCount` lanes. Tiling it as a register-wide masked read would (a)
      // index out of its small buffer with the loop IV and (b) degrade the
      // broadcast into an identity op, corrupting every lane but the first.
      // If the read also feeds a direct (non-broadcast) element-wise op, it
      // must be tiled to the register width instead: keeping it vector<1> would
      // mix a vector<1> operand with laneCount-wide operands (e.g. arith.mulf)
      // and fail verification. The broadcast consumers then become legal
      // identity broadcasts (vector<laneCountxT> -> vector<laneCountxT>).
      if (isUnitNPUVector(nvt) && onlyFeedsBroadcast(rd.getResult())) {
        SmallVector<int64_t> shape(nvt.getShape().begin(), nvt.getShape().end());
        auto unitVecTy = VectorType::get(shape, elemType);
        SmallVector<Value> unitIndices(rd.getIndices().size());
        std::transform(rd.getIndices().begin(), rd.getIndices().end(), unitIndices.begin(),
                       [&remap](Value idx) { return remap(idx); });
        auto unitPermMap = b.getMultiDimIdentityMap(unitVecTy.getRank());
        auto unitInBounds = b.getBoolArrayAttr(SmallVector<bool>(unitVecTy.getRank(), true));
        Value res = b.create<vector::TransferReadOp>(loc, unitVecTy, remap(rd.getSource()), unitIndices, unitPermMap,
                                                     remap(rd.getPadding()), /* mask= */ Value(), unitInBounds);
        map.map(rd.getResult(), res);
        return success();
      }
      int64_t srcRank = sourceRank(rd.getSource());
      Value res =
        b.create<vector::TransferReadOp>(loc, vecTypeOf(elemType), remap(rd.getSource()), trailingIndices(srcRank),
                                         permMapForRank(srcRank), remap(rd.getPadding()), mask, inBounds);
      map.map(rd.getResult(), res);
      return success();
    }
    if (auto wr = dyn_cast<npuv::TransferWriteOp>(op)) {
      int64_t srcRank = sourceRank(wr.getSource());
      b.create<vector::TransferWriteOp>(loc, /* resultType= */ Type(), remap(wr.getVector()), remap(wr.getSource()),
                                        trailingIndices(srcRank), permMapForRank(srcRank), mask, inBounds);
      return success();
    }
    if (auto bc = dyn_cast<npuv::BroadcastOp>(op)) {
      auto elemType = llvm::cast<npuv::NPUVectorType>(bc.getResult().getType()).getElementType();
      Value res = b.create<vector::BroadcastOp>(loc, vecTypeOf(elemType), remap(bc.getSource()));
      map.map(bc.getResult(), res);
      return success();
    }
    if (auto ci = dyn_cast<npuv::CmpIOp>(op)) {
      Value res = b.create<arith::CmpIOp>(loc, ci.getPredicate(), remap(ci.getLhs()), remap(ci.getRhs()));
      map.map(ci.getResult(), res);
      return success();
    }
    if (auto cf = dyn_cast<npuv::CmpFOp>(op)) {
      Value res = b.create<arith::CmpFOp>(loc, cf.getPredicate(), remap(cf.getLhs()), remap(cf.getRhs()));
      map.map(cf.getResult(), res);
      return success();
    }
    if (auto sel = dyn_cast<npuv::SelectOp>(op)) {
      Value res = b.create<arith::SelectOp>(loc, remap(sel.getCondition()), remap(sel.getTrueValue()),
                                            remap(sel.getFalseValue()));
      map.map(sel.getResult(), res);
      return success();
    }
    if (succeeded(tileUnaryCast(b, op, map, laneCount))) {
      return success();
    }
    if (llvm::isa_and_nonnull<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
      return tileArithMath(b, op, map, laneCount);
    }
    return op->emitError("npuvector-to-vector: unsupported op inside tileable kernel");
  }

  // Lower npuvector cast ops (extf/truncf/.../index_castui) to the matching
  // arith cast at `laneCount` width.
  LogicalResult tileUnaryCast(OpBuilder &b, Operation *op, IRMapping &map, int64_t laneCount) const {
    Location loc = op->getLoc();
    auto outElem = [](Operation *o) {
      return llvm::cast<npuv::NPUVectorType>(o->getResult(0).getType()).getElementType();
    };
    auto castTo = [&map, laneCount, &outElem](Operation *o, auto creator) -> LogicalResult {
      Value res = creator(VectorType::get({laneCount}, outElem(o)));
      map.map(o->getResult(0), res);
      return success();
    };

    if (auto c = dyn_cast<npuv::ExtFOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::ExtFOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::TruncFOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::TruncFOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::ExtSIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::ExtSIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::ExtUIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::ExtUIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::TruncIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::TruncIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::SIToFPOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::SIToFPOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::UIToFPOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::UIToFPOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::FPToSIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::FPToSIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::FPToUIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::FPToUIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::BitcastOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::BitcastOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::IndexCastOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::IndexCastOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    if (auto c = dyn_cast<npuv::IndexCastUIOp>(op)) {
      return castTo(op, [&b, loc, &map, &c](Type ty) {
        return b.create<arith::IndexCastUIOp>(loc, ty, map.lookupOrDefault(c.getIn())).getResult();
      });
    }
    return failure();
  }

  // Rebuild a generic arith/math op (binary elementwise, dense constant, ...)
  // with `laneCount`-wide vector operand/result types.
  LogicalResult tileArithMath(OpBuilder &b, Operation *op, IRMapping &map, int64_t laneCount) const {
    Location loc = op->getLoc();
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(cst.getType());
      if (!nvt) {
        Operation *cloned = b.clone(*op);
        map.map(cst.getResult(), cloned->getResult(0));
        return success();
      }
      auto vecType = VectorType::get({laneCount}, nvt.getElementType());
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat()) {
        return op->emitError("npuvector-to-vector: only splat npuvector constants are supported");
      }
      Value res = b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vecType, dense.getSplatValue<Attribute>()));
      map.map(cst.getResult(), res);
      return success();
    }

    SmallVector<Type> resultTypes;
    for (Type t : op->getResultTypes()) {
      if (auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(t)) {
        resultTypes.push_back(VectorType::get({laneCount}, nvt.getElementType()));
      } else {
        resultTypes.push_back(t);
      }
    }
    SmallVector<Value> operands(op->getNumOperands());
    std::transform(op->operand_begin(), op->operand_end(), operands.begin(),
                   [&map](Value v) { return map.lookupOrDefault(v); });
    OperationState state(loc, op->getName().getStringRef());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    Operation *newOp = b.create(state);
    for (auto [oldRes, newRes] : llvm::zip(op->getResults(), newOp->getResults())) {
      map.map(oldRes, newRes);
    }
    return success();
  }

  LogicalResult tileKernel(func::FuncOp vfFunc, int64_t laneCount) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();

    // Anchor the tile shape on a transfer_read (or transfer_write): its rank
    // gives the loop-nest depth and its static dims / dynamicSizes the bounds.
    npuv::NPUVectorType anchorType;
    SmallVector<Value> anchorDynSizes;
    Value anchorSource;
    if (failed(findTileAnchor(entry, anchorType, anchorDynSizes, anchorSource))) {
      return failure();
    }
    int64_t rank = anchorType.getRank();
    if (rank == 0) {
      return failure();
    }

    // The compute chain to move into the loop (anything touching npuvector).
    SmallVector<Operation *> computeOps;
    for (Operation &op : entry.without_terminator()) {
      if (touchesNpuVector(&op)) {
        computeOps.push_back(&op);
      }
    }
    if (computeOps.empty()) {
      return success();
    }

    OpBuilder builder(entry.getTerminator());
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cLane = builder.create<arith::ConstantIndexOp>(loc, laneCount);

    // Per-dimension upper bound: static extent -> constant, dynamic -> the
    // matching dynamicSizes operand (consumed in shape order).
    SmallVector<Value> bounds;
    bounds.reserve(rank);
    TileDesc anchorDesc{anchorType, anchorDynSizes, anchorSource};
    if (failed(collectTileBounds(builder, loc, anchorDesc, bounds))) {
      return failure();
    }

    // Build the loop nest: outer dims iterate one element at a time, the
    // innermost (register) dim iterates `laneCount` lanes at a time.
    SmallVector<Value> ivs;
    ivs.reserve(rank);
    OpBuilder bodyBuilder = builder;
    for (int64_t d = 0; d < rank - 1; ++d) {
      auto forOp = bodyBuilder.create<scf::ForOp>(loc, c0, bounds[d], c1);
      ivs.push_back(forOp.getInductionVar());
      bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
    }
    auto innerFor = bodyBuilder.create<scf::ForOp>(loc, c0, bounds[rank - 1], cLane);
    ivs.push_back(innerFor.getInductionVar());
    bodyBuilder = OpBuilder::atBlockBegin(innerFor.getBody());

    // Active lanes of the current innermost iteration: min(laneCount, ub - iv).
    Value remaining = bodyBuilder.create<arith::SubIOp>(loc, bounds[rank - 1], ivs.back());
    Value activeLen = bodyBuilder.create<arith::MinSIOp>(loc, remaining, cLane);
    auto maskType = VectorType::get({laneCount}, bodyBuilder.getI1Type());
    Value mask = bodyBuilder.create<vector::CreateMaskOp>(loc, maskType, ValueRange{activeLen});

    IRMapping map;
    if (std::any_of(computeOps.begin(), computeOps.end(),
                    [this, &bodyBuilder, &map, &ivs, &mask, laneCount](Operation *op) {
                      return failed(tileComputeOp(bodyBuilder, op, map, ivs, mask, laneCount));
                    })) {
      return failure();
    }

    for (auto it = computeOps.rbegin(); it != computeOps.rend(); ++it) {
      (*it)->erase();
    }

    // Drop constants that became dead once the npuvector ops were removed
    // (e.g. the old [index]/[maxSizes] constants).
    SmallVector<Operation *> deadConsts;
    for (Operation &op : entry.without_terminator()) {
      if (isa<arith::ConstantOp>(op) && op.use_empty()) {
        deadConsts.push_back(&op);
      }
    }
    for (Operation *op : deadConsts) {
      op->erase();
    }
    return success();
  }

  // Resolve the per-dimension loop bounds for a tile of type `type`: static
  // extents become index constants; dynamic extents take the matching
  // `dynSizes` operand (consumed in shape order, e.g. from transfer_read) and,
  // failing that, a `memref.dim` of the backing `source` memref. `builder`
  // inserts the freshly created ops.
  static LogicalResult collectTileBounds(OpBuilder &builder, Location loc, TileDesc desc,
                                         SmallVectorImpl<Value> &bounds) {
    ArrayRef<int64_t> shape = desc.type.getShape();
    unsigned dynIdx = 0;
    for (int64_t d = 0, rank = desc.type.getRank(); d < rank; ++d) {
      if (!ShapedType::isDynamic(shape[d])) {
        bounds.push_back(builder.create<arith::ConstantIndexOp>(loc, shape[d]));
      } else if (dynIdx < desc.dynSizes.size()) {
        bounds.push_back(desc.dynSizes[dynIdx++]);
      } else if (desc.source && llvm::isa<MemRefType>(desc.source.getType())) {
        bounds.push_back(builder.create<memref::DimOp>(loc, desc.source, d));
      } else {
        return failure();
      }
    }
    return success();
  }

  // Anchor the tile shape: prefer a transfer_read (carries dynamicSizes), else
  // fall back to a transfer_write whose stored value is an npuvector.
  static LogicalResult findTileAnchor(Block &entry, npuv::NPUVectorType &type, SmallVectorImpl<Value> &dynSizes,
                                      Value &source) {
    npuv::TransferReadOp refRead;
    npuv::TransferReadOp unitRead;
    npuv::TransferWriteOp refWrite;
    int64_t refReadRank = -1;
    for (Operation &op : entry) {
      if (auto rd = dyn_cast<npuv::TransferReadOp>(&op)) {
        // Unit reads are scalar / broadcast-source loads, never the iteration
        // space; only fall back to one if the kernel offers nothing better.
        if (isUnitNPUVector(rd.getVector().getType())) {
          if (!unitRead) {
            unitRead = rd;
          }
          continue;
        }
        // Anchor on the highest-rank read: it spans the full iteration space.
        // A lower-rank read (e.g. a 1-D bias broadcast onto a 2-D tensor) would
        // under-count the loop nest and leave higher-rank reads/writes with too
        // few indices.
        int64_t rank = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType()).getRank();
        if (!refRead || rank > refReadRank) {
          refRead = rd;
          refReadRank = rank;
        }
      } else if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        if (!refWrite && isNpuVectorType(wr.getVector().getType())) {
          refWrite = wr;
        }
      }
    }
    if (refRead) {
      type = llvm::cast<npuv::NPUVectorType>(refRead.getVector().getType());
      dynSizes.assign(refRead.getDynamicSizes().begin(), refRead.getDynamicSizes().end());
      source = refRead.getSource();
      return success();
    }
    if (refWrite) {
      type = llvm::cast<npuv::NPUVectorType>(refWrite.getVector().getType());
      source = refWrite.getSource();
      return success();
    }
    if (unitRead) {
      type = llvm::cast<npuv::NPUVectorType>(unitRead.getVector().getType());
      dynSizes.assign(unitRead.getDynamicSizes().begin(), unitRead.getDynamicSizes().end());
      source = unitRead.getSource();
      return success();
    }
    return failure();
  }

  // Identify the reduction anchor and the ops that feed it:
  //   - `redOp`   : the single reduction being tiled.
  //   - `refRead` : the read supplying the loop extent. Scalar / broadcast-source
  //                 unit reads are skipped in favour of a real read; a unit read
  //                 is used only as a fallback.
  //   - `producers`: the element-wise ops before the reduction that move into the
  //                 loop body.
  // Returns failure when there is no usable rank>0 anchor.
  static LogicalResult analyzeReductionKernel(func::FuncOp vfFunc, npuv::ReductionOp &redOp,
                                              npuv::TransferReadOp &refRead, SmallVector<Operation *> &producers) {
    Block &entry = vfFunc.getBody().front();
    npuv::TransferReadOp unitRead;
    for (Operation &op : entry) {
      if (auto r = dyn_cast<npuv::ReductionOp>(&op)) {
        redOp = r;
      }
      if (auto rd = dyn_cast<npuv::TransferReadOp>(&op)) {
        if (isUnitNPUVector(rd.getVector().getType())) {
          if (!unitRead) {
            unitRead = rd;
          }
        } else if (!refRead) {
          refRead = rd;
        }
      }
    }
    if (!refRead) {
      refRead = unitRead;
    }
    if (!redOp || !refRead) {
      return failure();
    }
    if (llvm::cast<npuv::NPUVectorType>(refRead.getVector().getType()).getRank() == 0) {
      return failure();
    }
    // Producers (element-wise ops before the reduction) move into the loop.
    for (Operation &op : entry.without_terminator()) {
      if (&op == redOp.getOperation()) {
        break;
      }
      if (touchesNpuVector(&op)) {
        producers.push_back(&op);
      }
    }
    return success();
  }

  // Locate the result memref this reduction writes into (the output buffer the
  // outlining provides, e.g. `memref<1xf32>`), tracing the scalar result to its
  // memref.store. That buffer doubles as the seed/accumulator store: storing
  // then reading the identity back makes the multi_reduction acc a real
  // (non-foldable) memory value, which is what lets NormalizeVector legalize the
  // op. Fall back to a scratch alloc when the kernel returns the scalar instead.
  // The seed (the op's existing acc or `idScalar`) is stored into the buffer and
  // the delivering store is reported back so the caller can drop it later.
  static Value prepareAccumulatorBuffer(OpBuilder &builder, Location loc, ReductionDesc desc,
                                        memref::StoreOp &deliveryStore) {
    Value outBuf;
    for (Operation *user : desc.redOp.getResult().getUsers()) {
      if (auto st = dyn_cast<memref::StoreOp>(user)) {
        auto mt = llvm::dyn_cast<MemRefType>(st.getMemRef().getType());
        if (mt && mt.getRank() == 1 && mt.getElementType() == desc.scalarElem) {
          outBuf = st.getMemRef();
          deliveryStore = st;
          break;
        }
      }
    }

    Value seedInit = desc.redOp.getAcc() ? desc.redOp.getAcc() : desc.idScalar;
    Value accBuf = outBuf;
    if (!accBuf) {
      auto seedMemTy = MemRefType::get({1}, desc.scalarElem);
      accBuf = builder.create<memref::AllocOp>(loc, seedMemTy);
    }
    builder.create<memref::StoreOp>(loc, seedInit, accBuf, ValueRange{desc.c0});
    return accBuf;
  }

  // Tiled full-reduction for performance and NormalizeVector compatibility:
  //   - Inside scf.for: element-wise style accumulation (iter_args holding
  //     vector<laneCountxT>, lane-wise combine via arith ops + mask select).
  //   - After the loop: one horizontal vector.multi_reduction of the register
  //     tile to a scalar, with a scalar acc materialised via
  //     extractelement(rank-0 read of a 1-element seed buffer) and its result
  //     fed to a vector.broadcast — the shape NormalizeVector lowers to HIVM.
  LogicalResult tileReductionKernel(func::FuncOp vfFunc, int64_t laneCount) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();

    npuv::ReductionOp redOp;
    npuv::TransferReadOp refRead;
    SmallVector<Operation *> producers;
    if (failed(analyzeReductionKernel(vfFunc, redOp, refRead, producers))) {
      return failure();
    }
    auto nvt = llvm::cast<npuv::NPUVectorType>(refRead.getVector().getType());
    int64_t rank = nvt.getRank();

    auto kind = redOp.getKind();
    Type scalarElem = getElementTypeOrSelf(redOp.getDest().getType());
    auto regVecTy = VectorType::get({laneCount}, scalarElem);

    OpBuilder builder(redOp);
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cLane = builder.create<arith::ConstantIndexOp>(loc, laneCount);

    SmallVector<Value> bounds;
    bounds.reserve(rank);
    TileDesc readDesc{nvt, refRead.getDynamicSizes(), refRead.getSource()};
    if (failed(collectTileBounds(builder, loc, readDesc, bounds))) {
      return failure();
    }

    // Register-wide identity for the element-wise loop accumulator.
    Value regIdent = buildIdentityAccumulator(builder, loc, kind, regVecTy);

    TypedAttr idAttr = getCombiningIdentityAttr(builder, kind, scalarElem);
    Value idScalar = builder.create<arith::ConstantOp>(loc, idAttr);
    memref::StoreOp deliveryStore;
    ReductionDesc redDesc{redOp, scalarElem, idScalar, c0};
    Value accBuf = prepareAccumulatorBuffer(builder, loc, redDesc, deliveryStore);

    // Build the loop nest with a vector<laneCountxT> iter_arg threaded through
    // every level; outer dims step 1, innermost step laneCount.
    SmallVector<Value> ivs(rank);
    SmallVector<scf::ForOp> loops(rank);
    OpBuilder lb = builder;
    Value curInit = regIdent;
    for (int64_t d = 0; d < rank; ++d) {
      Value step = (d == rank - 1) ? cLane : c1;
      auto forOp = lb.create<scf::ForOp>(loc, c0, bounds[d], step, ValueRange{curInit});
      loops[d] = forOp;
      ivs[d] = forOp.getInductionVar();
      curInit = forOp.getRegionIterArgs().front();
      lb = OpBuilder::atBlockBegin(forOp.getBody());
    }

    // Innermost body: mask, replay producers, lane-wise accumulate.
    Value remaining = lb.create<arith::SubIOp>(loc, bounds[rank - 1], ivs.back());
    Value activeLen = lb.create<arith::MinSIOp>(loc, remaining, cLane);
    auto tileMaskTy = VectorType::get({laneCount}, lb.getI1Type());
    Value tileMask = lb.create<vector::CreateMaskOp>(loc, tileMaskTy, ValueRange{activeLen});

    IRMapping map;
    if (std::any_of(producers.begin(), producers.end(), [this, &lb, &map, &ivs, &tileMask, laneCount](Operation *op) {
          return failed(tileComputeOp(lb, op, map, ivs, tileMask, laneCount));
        })) {
      return failure();
    }
    Value redInput = map.lookupOrDefault(redOp.getVector());
    if (redInput.getType() != regVecTy) {
      return redOp.emitError("npuvector-to-vector: reduction input register type mismatch");
    }
    // Inactive lanes carry the identity so they do not affect the accumulator.
    Value masked = lb.create<arith::SelectOp>(loc, tileMask, redInput, regIdent);
    Value innerAcc = loops[rank - 1].getRegionIterArgs().front();
    Value combined = vector::makeArithReduction(lb, loc, kind, innerAcc, masked);
    lb.create<scf::YieldOp>(loc, combined);

    // Yield the inner result up through enclosing loops.
    for (int64_t d = rank - 2; d >= 0; --d) {
      OpBuilder yb = OpBuilder::atBlockEnd(loops[d].getBody());
      yb.create<scf::YieldOp>(loc, loops[d + 1].getResult(0));
    }

    // After the loop: reduce the register tile to a scalar in the exact shape
    // NormalizeVector's UnitDimMultiReductionToReduction expects:
    //   %seed = vector.extractelement (rank-0 transfer_read of accBuf)
    //   %r    = vector.multi_reduction <kind>, %vecAcc, %seed [0] : vector<NxT> to T
    //   %bc   = vector.broadcast %r : T to vector<1xT>
    //   vector.transfer_write %bc, %accBuf[0]   (keeps the broadcast consumer live)
    // The acc MUST come from an extractelement and the result MUST feed a
    // broadcast, otherwise the pattern bails and the op stays illegal.
    Value vecAcc = loops[0].getResult(0);
    OpBuilder post(redOp);
    auto rank0Ty = VectorType::get({}, scalarElem);
    auto rank0Map = AffineMap::get(/* dimCount= */ 1, /* symbolCount= */ 0, post.getContext());
    Value seedVec = post.create<vector::TransferReadOp>(loc, rank0Ty, accBuf, ValueRange{c0}, rank0Map, idScalar,
                                                        /* mask= */ Value(), post.getBoolArrayAttr({}));
    Value seed = post.create<vector::ExtractElementOp>(loc, seedVec);
    SmallVector<bool> reductionMask(1, true);
    Value result = post.create<vector::MultiDimReductionOp>(loc, vecAcc, seed, reductionMask, kind);

    // Route the scalar to existing consumers first, THEN add the broadcast so it
    // is the first user the NormalizeVector pattern inspects.
    redOp.getResult().replaceAllUsesWith(result);
    auto out1Ty = VectorType::get({1}, scalarElem);
    Value bcast = post.create<vector::BroadcastOp>(loc, out1Ty, result);
    Value oneMask = post.create<vector::CreateMaskOp>(loc, VectorType::get({1}, post.getI1Type()), ValueRange{c1});
    post.create<vector::TransferWriteOp>(loc, /* resultType= */ Type(), bcast, accBuf, ValueRange{c0},
                                         post.getMultiDimIdentityMap(1), oneMask, post.getBoolArrayAttr({true}));

    // Drop the original scalar store: the broadcast + transfer_write above is now
    // the canonical write into the output buffer (avoids a double store).
    if (deliveryStore) {
      deliveryStore.erase();
    }
    redOp.erase();
    for (auto it = producers.rbegin(); it != producers.rend(); ++it) {
      (*it)->erase();
    }

    SmallVector<Operation *> deadConsts;
    for (Operation &op : entry.without_terminator()) {
      if (isa<arith::ConstantOp>(op) && op.use_empty()) {
        deadConsts.push_back(&op);
      }
    }
    for (Operation *op : deadConsts) {
      op->erase();
    }
    return success();
  }

  // -------------------------------------------------------------------------
  // Fallback path: direct per-op type conversion (no register tiling).
  // -------------------------------------------------------------------------
  LogicalResult directConvert(func::FuncOp vfFunc, int64_t laneCount) {
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    typeConverter.addConversion(
      [laneCount](npuv::NPUVectorType t) -> Type { return convertToVectorType(t, laneCount); });

    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addLegalDialect<vector::VectorDialect, memref::MemRefDialect, func::FuncDialect, scf::SCFDialect>();
    target.addIllegalDialect<npuv::NPUVectorDialect>();
    auto onlyVectorTypes = [&typeConverter](Operation *op) { return typeConverter.isLegal(op); };
    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(onlyVectorTypes);

    RewritePatternSet patterns(ctx);
    patterns.add<TransferReadLowering, TransferWriteLowering, ReductionLowering, BroadcastLowering, TransposeLowering,
                 CmpILowering, CmpFLowering, SelectLowering>(typeConverter, ctx);
    patterns
      .add<UnaryCastLowering<npuv::ExtFOp, arith::ExtFOp>, UnaryCastLowering<npuv::TruncFOp, arith::TruncFOp>,
           UnaryCastLowering<npuv::ExtSIOp, arith::ExtSIOp>, UnaryCastLowering<npuv::ExtUIOp, arith::ExtUIOp>,
           UnaryCastLowering<npuv::TruncIOp, arith::TruncIOp>, UnaryCastLowering<npuv::SIToFPOp, arith::SIToFPOp>,
           UnaryCastLowering<npuv::UIToFPOp, arith::UIToFPOp>, UnaryCastLowering<npuv::FPToSIOp, arith::FPToSIOp>,
           UnaryCastLowering<npuv::FPToUIOp, arith::FPToUIOp>, UnaryCastLowering<npuv::BitcastOp, arith::BitcastOp>,
           UnaryCastLowering<npuv::IndexCastOp, arith::IndexCastOp>,
           UnaryCastLowering<npuv::IndexCastUIOp, arith::IndexCastUIOp>>(typeConverter, ctx);
    patterns.add<ElementwiseRetypeLowering>(typeConverter, ctx);

    return applyPartialConversion(vfFunc, target, std::move(patterns));
  }

  LogicalResult convertKernel(func::FuncOp vfFunc) {
    func::FuncOp archFunc = findArchFunc(vfFunc);
    std::optional<int64_t> regWidth = getRegisterWidthBytes(archFunc);
    if (!regWidth) {
      vfFunc.emitError(
        "cannot determine register vector width: missing 'arch' attribute on the kernel "
        "or any of its callers");
      return failure();
    }
    int64_t laneCount = computeLaneCount(vfFunc, *regWidth);

    if (isTileable(vfFunc)) {
      return tileKernel(vfFunc, laneCount);
    }
    if (isReductionTileable(vfFunc)) {
      return tileReductionKernel(vfFunc, laneCount);
    }
    return directConvert(vfFunc, laneCount);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: collect the functions tagged as vector kernels.
    SmallVector<func::FuncOp> vfFuncs;
    module.walk([&vfFuncs](func::FuncOp f) {
      if (f->hasAttr(kVectorFunctionAttr)) {
        vfFuncs.push_back(f);
      }
    });

    // Steps 2 & 3: lower npuvector to community vector inside each kernel.
    if (std::any_of(vfFuncs.begin(), vfFuncs.end(), [this](func::FuncOp f) { return failed(convertKernel(f)); })) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createNPUVectorToVectorPass() { return std::make_unique<NPUVectorToVector>(); }
}  // namespace mlir
