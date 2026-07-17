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

// =============================================================================
// NPUVectorToVector
//
// This pass lowers the `npuvector` ops inside every outlined vector function
// (`vf`) to community `vector` ops, **and** guarantees that after the pass every
// op in the vf operates on at most a single physical register worth of data
// (otherwise the generated kernel blows the stack).
//
// To keep the two responsibilities separate (they used to be intertwined, which
// made every new corner case harder than the last) the pass runs two
// independent steps per vf, both visible in `runOnOperation`:
//
//   Step 1 - convertNpuVectorToVector():
//     Pure analysis / planning. It computes
//       * the lane count for the kernel's widest data type (how many elements
//         fit in one register), and
//       * for every op the number of `scf.for` layers required to fit the
//         register, plus the kernel "category" (elementwise / full-reduction /
//         partial-reduction / transpose).
//     It also fixes the lowering contract used by step 2: reduction lowers to
//     `vector.multi_reduction` (unchanged from before), and broadcast must honor
//     the npuvector broadcast `dimension` mapping.
//
//   Step 2 - tileVectorToRegister():
//     Uses the plan from step 1 to emit the register-sized `scf.for` nest and
//     the community `vector` ops inside it. Reduction keeps the previous
//     accumulate-then-horizontal-reduce shape; broadcast and transpose are
//     re-implemented so the per-iteration tiles always fit a register.
//
// =============================================================================

#include "akg/Conversion/NPUVectorToVector/NPUVectorToVector.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_DEF_NPUVECTORTOVECTOR
#include "akg/Conversion/Passes.h.inc"

namespace {

namespace npuv = mlir::npuvector;

// Inline element count for small dense sets of reduction dims / axes.
constexpr unsigned kSmallDenseSetSize = 8;

// ===========================================================================
// Register-width helpers
// ===========================================================================

// Get the register vector width (in bytes) of the device described by `func`'s
// `arch` attribute.
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
  ModuleOp module = vfFunc->getParentOfType<ModuleOp>();
  if (!module) {
    return func::FuncOp();
  }
  func::FuncOp found;
  module.walk([&](func::CallOp call) {
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
  if (bitWidth < CHAR_BIT) {
    return 0;
  }
  return bitWidth / CHAR_BIT;
}

// Compute one lane count for the whole vf kernel so that every register fits the
// physical width: `regWidthBytes / maxElemBytes`. Sizing for the *widest*
// (sub-byte excluded) data type guarantees no register overflows. With a 256B
// register: pure f32 -> 64 lanes; pure f16 -> 128 lanes; mixing i64 (8B) and i8
// -> 32 lanes (both within budget).
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
  vfFunc.walk([&](Operation *op) {
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
// `!npuvector<1xf32>`). Such values are scalar / broadcast source loads that are
// loop invariant: they must be read at their own fixed index and kept tiny (NOT
// tiled to register width with the loop IV, which would read out of bounds of
// their small backing buffer -- the precision bug).
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

static int64_t npuVectorRank(Type t) {
  auto vt = llvm::dyn_cast_or_null<npuv::NPUVectorType>(t);
  return vt ? vt.getRank() : 0;
}

// Convert an `!npuvector` type to a community `vector` type, resolving every
// dynamic extent to `laneCount`. Used by the directConvert fallback and as the
// canonical npuvector->vector shape mapping.
// Rank-0 (scalar) npuvector -- the "npuvector+scalar" form -- maps to
// `vector<1xT>`, NOT a shapeless rank-0 `vector<T>`. Every lowered vector must
// carry an explicit shape so downstream passes never see 0-D vectors.
static VectorType shapedVectorType(ArrayRef<int64_t> shape, Type elemType) {
  if (shape.empty()) {
    return VectorType::get({1}, elemType);
  }
  return VectorType::get(shape, elemType);
}

static VectorType convertToVectorType(npuv::NPUVectorType vt, int64_t laneCount) {
  SmallVector<int64_t> shape(vt.getShape().begin(), vt.getShape().end());
  for (int64_t &d : shape) {
    if (ShapedType::isDynamic(d)) {
      d = laneCount;
    }
  }
  return shapedVectorType(shape, vt.getElementType());
}

// The rank of a `memref`/`tensor` value; -1 if `src` is not a shaped type.
static int64_t getShapedSourceRank(Value src) {
  if (auto st = llvm::dyn_cast<ShapedType>(src.getType())) {
    return st.getRank();
  }
  return -1;
}

// Drop leading no-op indices when the npuvector transfer op carries more
// indices than the source memref's rank. The upstream may now emit a
// placeholder `[%c0]` even for rank-0 memrefs (the "npuvector+scalar" form):
// the conversion treats those leading entries as 0 and discards them so the
// generated `vector.transfer_*` op matches the source shape.
static SmallVector<Value> clampIndicesToSourceRank(ValueRange indices, int64_t srcRank) {
  SmallVector<Value> result(indices.begin(), indices.end());
  if (srcRank >= 0 && static_cast<int64_t>(result.size()) > srcRank) {
    result.erase(result.begin(), result.begin() + (result.size() - srcRank));
  }
  return result;
}

// Build a broadcast-style permutation map for a `vector.transfer_read` whose
// source is rank 0: every position of the rank-`vecRank` result vector reads
// the single source element. Emits `affine_map<() -> (0, ..., 0)>`.
static AffineMap getRank0BroadcastPermMap(OpBuilder &b, unsigned vecRank) {
  SmallVector<AffineExpr> exprs(vecRank, b.getAffineConstantExpr(0));
  return AffineMap::get(0, 0, exprs, b.getContext());
}

// Write a (possibly register-sized) vector's leading lane back to a rank-0
// `memref` using a rank-0 `vector.transfer_write` (never `memref.store`, which
// the downstream lowering does not accept): extract the scalar lane, wrap it
// in a rank-0 vector and transfer_write it with zero indices and the rank-0
// permutation map `() -> ()`.
static void emitRank0TransferWrite(OpBuilder &b, Location loc, Value vecVal, Value memref) {
  auto vTy = llvm::cast<VectorType>(vecVal.getType());
  Value scalar;
  if (vTy.getRank() == 0) {
    scalar = b.create<vector::ExtractOp>(loc, vecVal, ArrayRef<int64_t>{});
  } else {
    SmallVector<int64_t> zeros(vTy.getRank(), 0);
    scalar = b.create<vector::ExtractOp>(loc, vecVal, zeros);
  }
  auto rank0Ty = VectorType::get({}, vTy.getElementType());
  Value rank0Vec = b.create<vector::BroadcastOp>(loc, rank0Ty, scalar);
  auto rank0Map = AffineMap::get(0, 0, b.getContext());
  b.create<vector::TransferWriteOp>(loc, Type(), rank0Vec, memref, ValueRange{}, rank0Map,
                                    Value(), b.getBoolArrayAttr({}));
}

// ===========================================================================
// Reduction identity helpers
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
        return FloatAttr::get(elemType, APFloat::getInf(sem, false));
      case CK::MAXNUMF:
      case CK::MAXIMUMF:
        return FloatAttr::get(elemType, APFloat::getInf(sem, true));
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
      return b.getIntegerAttr(elemType, 0);
  }
}

static Value buildIdentityAccumulator(OpBuilder &b, Location loc, vector::CombiningKind kind, Type destType) {
  Type elemType = getElementTypeOrSelf(destType);
  TypedAttr scalarAttr = getCombiningIdentityAttr(b, kind, elemType);
  if (auto vt = llvm::dyn_cast<VectorType>(destType)) {
    return b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vt, scalarAttr));
  }
  return b.create<arith::ConstantOp>(loc, scalarAttr);
}

// The npuvector broadcast `dimension` mapping: m[i] is the destination dim that
// source dim i lands on. An empty attribute means the prefix mapping m[i]=i.
static SmallVector<int64_t> getBroadcastDimMap(npuv::BroadcastOp bc) {
  SmallVector<int64_t> m;
  ArrayRef<int64_t> dim = bc.getDimension();
  int64_t srcRank = npuVectorRank(bc.getSource().getType());
  if (!dim.empty()) {
    m.assign(dim.begin(), dim.end());
  } else {
    for (int64_t i = 0; i < srcRank; ++i) {
      m.push_back(i);
    }
  }
  return m;
}

// ===========================================================================
// Fallback lowering patterns (directConvert path, used for kernels that do not
// match any tileable category).
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
    int64_t srcRank = getShapedSourceRank(adaptor.getSource());
    if (srcRank < 0) {
      return rewriter.notifyMatchFailure(op, "transfer_read source is not a shaped type");
    }
    SmallVector<Value> indices = clampIndicesToSourceRank(adaptor.getIndices(), srcRank);
    // For a rank-0 source memref with a rank>0 result vector, every lane reads
    // the single source element: emit a broadcast permutation map. Otherwise
    // use the minor-identity map that matches the source/result rank pair.
    AffineMap permMap =
      (srcRank == 0 && vecType.getRank() > 0)
        ? getRank0BroadcastPermMap(rewriter, vecType.getRank())
        : AffineMap::getMinorIdentityMap(static_cast<unsigned>(srcRank), vecType.getRank(), rewriter.getContext());
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(vecType.getRank(), true));
    auto newRead = rewriter.create<vector::TransferReadOp>(op.getLoc(), vecType, adaptor.getSource(), indices, permMap,
                                                           adaptor.getPadding(),
                                                           Value(), inBounds);
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
    int64_t srcRank = getShapedSourceRank(adaptor.getSource());
    if (srcRank < 0) {
      return rewriter.notifyMatchFailure(op, "transfer_write source is not a shaped type");
    }
    SmallVector<Value> indices = clampIndicesToSourceRank(adaptor.getIndices(), srcRank);
    // Writing a rank>0 vector to a rank-0 memref: the destination only holds a
    // single element. Extract the leading vector lane (the kernel is expected
    // to have computed a broadcast scalar) and write it back with a rank-0
    if (srcRank == 0 && vecType.getRank() > 0) {
      emitRank0TransferWrite(rewriter, op.getLoc(), adaptor.getVector(), adaptor.getSource());
      rewriter.eraseOp(op);
      return success();
    }
    auto permMap = AffineMapAttr::get(
      AffineMap::getMinorIdentityMap(static_cast<unsigned>(srcRank), vecType.getRank(), rewriter.getContext()));
    auto inBounds = rewriter.getBoolArrayAttr(SmallVector<bool>(vecType.getRank(), true));
    rewriter.create<vector::TransferWriteOp>(op.getLoc(), adaptor.getVector(), adaptor.getSource(), indices, permMap,
                                             inBounds);
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
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 0, ctx) {}

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
// KernelPlan -- produced by step 1, consumed by step 2.
// ===========================================================================

enum class KernelCategory { Elementwise, FullReduction, PartialReduction, Transpose, Fallback };

struct KernelPlan {
  int64_t laneCount = 1;
  KernelCategory category = KernelCategory::Fallback;
  // Number of scf.for layers the kernel needs to fit a register (the iteration
  // space rank). The innermost layer is register-tiled (step = laneCount).
  int64_t loopRank = 0;
  // Per-op loop layers metadata (the rank of each op's tile). Used to document
  // the plan; step 2 derives the concrete tiles from the npuvector types.
  llvm::DenseMap<Operation *, int64_t> opLayers;
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
    registry.insert<affine::AffineDialect, arith::ArithDialect, math::MathDialect, npuv::NPUVectorDialect,
                    vector::VectorDialect, memref::MemRefDialect, func::FuncDialect, scf::SCFDialect>();
  }

  // -------------------------------------------------------------------------
  // Small shared helpers used by the tilers.
  // -------------------------------------------------------------------------

  static bool isConstantZeroIndex(Value v) {
    auto cst = v.getDefiningOp<arith::ConstantIndexOp>();
    return cst && cst.value() == 0;
  }

  // The constant value of an index `Value`, when it is an `arith.constant`.
  static std::optional<int64_t> getConstantIndex(Value v) {
    if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>()) {
      return cst.value();
    }
    return std::nullopt;
  }

  // index = base + iv (folds base==0).
  static Value addIndex(OpBuilder &b, Location loc, Value base, Value iv) {
    if (!iv) {
      return base;
    }
    if (!base || isConstantZeroIndex(base)) {
      return iv;
    }
    return b.create<arith::AddIOp>(loc, base, iv);
  }

  // -------------------------------------------------------------------------
  // Register-tile loop splitting (main aligned loop + masked tail loop)
  // The upstream pass aligns elementwise/transpose extents to the lane count,
  // but a reduction tile may be an arbitrary length (e.g. a `!npuvector<16xf32>`
  // reduced with 64 lanes). Reading/accumulating a full register there would
  // pull in out-of-range lanes and corrupt the result. To keep the common
  // aligned case fast we split the innermost (register) loop in two:
  //   * a main loop over [0, alignedBound) with full, unmasked tiles, and
  //   * a tail loop over [alignedBound, bound) (at most one trip) whose lanes
  //     are masked to the < lane-count remainder.
  // The split uses affine maps for the bounds and relies on the tail loop's
  // natural 0/1 trip count instead of an `scf.if`.
  // -------------------------------------------------------------------------

  // alignedBound = (bound floordiv lanes) * lanes, via an affine map.
  Value alignedRegBound(OpBuilder &b, Location loc, Value bound, int64_t lanes) const {
    AffineExpr d0 = b.getAffineDimExpr(0);
    AffineMap m = AffineMap::get(1, 0, d0.floorDiv(lanes) * lanes, b.getContext());
    return b.create<affine::AffineApplyOp>(loc, m, ValueRange{bound});
  }

  // Mask for the tail iteration: the active lane count is `bound - iv` (always
  // < lanes because the tail range is shorter than one register).
  Value tailRegMask(OpBuilder &b, Location loc, Value bound, Value iv, int64_t lanes) const {
    AffineExpr d0 = b.getAffineDimExpr(0);
    AffineExpr d1 = b.getAffineDimExpr(1);
    AffineMap m = AffineMap::get(2, 0, d0 - d1, b.getContext());
    Value len = b.create<affine::AffineApplyOp>(loc, m, ValueRange{bound, iv});
    return b.create<vector::CreateMaskOp>(loc, VectorType::get({lanes}, b.getI1Type()), ValueRange{len});
  }

  // Plan for splitting a register loop of extent `bound`. When `bound` is a
  // static multiple of `lanes` the tail is elided (preserving the fast aligned
  // lowering); when it is statically smaller than one register the main loop is
  // elided (a single masked loop). Dynamic extents emit both halves.
  struct RegSplit {
    bool emitMain = true;
    bool emitTail = true;
    Value alignedBound;  // lower bound of the tail / upper bound of the main loop
  };
  RegSplit planRegSplit(OpBuilder &b, Location loc, Value bound, int64_t lanes) const {
    RegSplit s;
    if (auto bc = getConstantIndex(bound)) {
      int64_t aligned = (*bc / lanes) * lanes;
      s.emitMain = aligned > 0;
      s.emitTail = aligned < *bc;
      if (aligned == *bc) {
        // Already aligned: reuse the extent so the fast path lowers
        // byte-identically (no redundant constant, tail elided).
        s.alignedBound = bound;
      } else if (aligned == 0) {
        // Shorter than one register: only the tail loop runs, starting at c0.
        s.alignedBound = Value();
      } else {
        s.alignedBound = b.create<arith::ConstantIndexOp>(loc, aligned);
      }
    } else {
      s.alignedBound = alignedRegBound(b, loc, bound, lanes);
    }
    return s;
  }

  // Per-dimension loop bounds derived from an anchor npuvector value. Static
  // extents become index constants; dynamic extents take the matching runtime
  // size from the producing transfer_read / broadcast.
  LogicalResult getPerDimBounds(OpBuilder &b, Location loc, Value anchorVal, SmallVectorImpl<Value> &bounds) const {
    auto vt = llvm::dyn_cast<npuv::NPUVectorType>(anchorVal.getType());
    if (!vt) {
      return failure();
    }
    ArrayRef<int64_t> shape = vt.getShape();
    int64_t rank = vt.getRank();
    unsigned numDyn = vt.getNumDynamicDims();

    // An extract_slice narrows the iteration to its sub-tile: the per-dim bound
    // of the result is the slice size of the corresponding kept source dim.
    if (auto es = anchorVal.getDefiningOp<npuv::ExtractSliceOp>()) {
      ArrayRef<int64_t> keep = es.getKeepDims();
      ValueRange sizes = es.getSizes();
      if (static_cast<int64_t>(keep.size()) != rank) {
        return failure();
      }
      for (int64_t r = 0; r < rank; ++r) {
        int64_t sd = keep[r];
        if (sd < 0 || sd >= static_cast<int64_t>(sizes.size())) {
          return failure();
        }
        bounds.push_back(sizes[sd]);
      }
      return success();
    }

    SmallVector<Value> dynSizes;
    Value source;
    if (auto rd = anchorVal.getDefiningOp<npuv::TransferReadOp>()) {
      dynSizes.assign(rd.getDynamicSizes().begin(), rd.getDynamicSizes().end());
      source = rd.getSource();
    } else if (auto bc = anchorVal.getDefiningOp<npuv::BroadcastOp>()) {
      dynSizes.assign(bc.getDynamicSizes().begin(), bc.getDynamicSizes().end());
    }

    // A broadcast lists one size per result dim; a transfer_read lists one per
    // dynamic dim only.
    if (static_cast<int64_t>(dynSizes.size()) == rank) {
      bounds.assign(dynSizes.begin(), dynSizes.end());
      return success();
    }

    unsigned dynIdx = 0;
    for (int64_t d = 0; d < rank; ++d) {
      if (!ShapedType::isDynamic(shape[d])) {
        bounds.push_back(b.create<arith::ConstantIndexOp>(loc, shape[d]));
      } else if (dynSizes.size() == numDyn && dynIdx < dynSizes.size()) {
        bounds.push_back(dynSizes[dynIdx++]);
      } else if (source && llvm::isa<MemRefType>(source.getType())) {
        bounds.push_back(b.create<memref::DimOp>(loc, source, d));
      } else {
        return failure();
      }
    }
    return success();
  }

  // -------------------------------------------------------------------------
  // Dimension mapping: per npuvector value, which loop dim each value dim maps
  // to. This is what makes broadcast-with-dimension and multi-rank kernels
  // correct.
  // -------------------------------------------------------------------------

  using DimMap = llvm::DenseMap<Value, SmallVector<int64_t>>;

  // Builder + location pair, used by most emission helpers.
  struct BuilderEnv {
    OpBuilder &builder;
    Location loc;
  };

  // Dim map + unit values bundle.
  struct DimMapState {
    DimMap &dimMap;
    llvm::DenseSet<Value> &unitVals;
  };

  // Per-loop-dim bounds, aggregated across every transfer in the kernel. Each
  // transfer value contributes its own per-dim extent to the loop dim it maps
  // to (via its dim map). This generalizes getPerDimBounds to permuted kernels
  // (transpose), where no single value is identity-aligned to the loop nest:
  // the contiguous write supplies the static extents while the read supplies
  // the dynamic ones, etc.
  LogicalResult getLoopBounds(BuilderEnv env, Block &entry, const DimMap &dimMap, int64_t loopRank,
                              SmallVectorImpl<Value> &loopBounds) const {
    OpBuilder &b = env.builder;
    Location loc = env.loc;
    loopBounds.assign(loopRank, Value());
    auto tryFill = [this, &b, &loc, &dimMap, &loopRank, &loopBounds](Value v) {
      auto it = dimMap.find(v);
      if (it == dimMap.end()) {
        return;
      }
      ArrayRef<int64_t> m = it->second;
      SmallVector<Value> ext;
      if (failed(getPerDimBounds(b, loc, v, ext)) || ext.size() != m.size()) {
        return;
      }
      for (size_t d = 0; d < m.size(); ++d) {
        int64_t ld = m[d];
        if (ld >= 0 && ld < loopRank && !loopBounds[ld]) {
          loopBounds[ld] = ext[d];
        }
      }
    };
    // A slice constrains the iteration to its sub-tile. Fill the sliced dims
    // from the extract_slice sizes first so a full-tile transfer_read (which
    // spans the padded, unsliced extent) cannot claim them with the wrong bound.
    auto feedsExtractSlice = [](Value v) {
      return llvm::any_of(v.getUsers(), [](Operation *u) { return isa<npuv::ExtractSliceOp>(u); });
    };
    for (Operation &op : entry) {
      if (auto es = dyn_cast<npuv::ExtractSliceOp>(&op)) {
        tryFill(es.getResult());
      }
    }
    for (Operation &op : entry) {
      if (auto rd = dyn_cast<npuv::TransferReadOp>(&op)) {
        if (feedsExtractSlice(rd.getResult())) {
          continue;
        }
        tryFill(rd.getResult());
      } else if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        if (wr.getVector().getDefiningOp<npuv::InsertSliceOp>()) {
          continue;
        }
        tryFill(wr.getVector());
      }
    }
    for (int64_t d = 0; d < loopRank; ++d) {
      if (!loopBounds[d]) {
        return failure();
      }
    }
    return success();
  }

  // One fixpoint relaxation step over `op`, deriving any newly-known maps.
  static bool relaxDimMap(Operation *op, int64_t loopRank, DimMap &dimMap, llvm::DenseSet<Value> &unitVals) {
    bool changed = false;
    auto known = [&dimMap](Value v) { return dimMap.count(v) != 0; };
    auto setMap = [&unitVals, &dimMap, &changed](Value v, ArrayRef<int64_t> m) {
      if (!isNpuVectorType(v.getType()) || unitVals.count(v) || dimMap.count(v)) {
        return;
      }
      dimMap[v] = SmallVector<int64_t>(m.begin(), m.end());
      changed = true;
    };

    if (auto bc = dyn_cast<npuv::BroadcastOp>(op)) {
      return relaxBroadcast(bc, dimMap, unitVals, known, setMap) || changed;
    }
    if (auto tr = dyn_cast<npuv::TransposeOp>(op)) {
      return relaxTranspose(tr, dimMap, known, setMap) || changed;
    }
    if (auto red = dyn_cast<npuv::ReductionOp>(op)) {
      return relaxReduction(red, dimMap, known, setMap) || changed;
    }
    if (auto es = dyn_cast<npuv::ExtractSliceOp>(op)) {
      return relaxExtractSlice(es, dimMap, known, setMap) || changed;
    }
    if (auto is = dyn_cast<npuv::InsertSliceOp>(op)) {
      return relaxInsertSlice(is, dimMap, known, setMap) || changed;
    }
    // Transfer writes are seeded separately and transfer-read result maps are
    // learned from consumers; neither has anything to relax here.
    if (isa<npuv::TransferWriteOp, npuv::TransferReadOp>(op)) {
      return changed;
    }
    relaxElementwiseDimMap(op, dimMap, unitVals, setMap);
    return changed;
  }

  // Relaxation helper for elementwise-like ops (arith/math/cmp/select/cast on
  // npuvector): every npuvector operand and result shares the same shape, hence
  // the same map.
  static void relaxElementwiseDimMap(Operation *op, DimMap &dimMap, const llvm::DenseSet<Value> &unitVals,
                                     const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    SmallVector<Value> vecVals;
    for (Value v : op->getOperands()) {
      if (isNpuVectorType(v.getType()) && !unitVals.count(v)) {
        vecVals.push_back(v);
      }
    }
    for (Value v : op->getResults()) {
      if (isNpuVectorType(v.getType()) && !unitVals.count(v)) {
        vecVals.push_back(v);
      }
    }
    SmallVector<int64_t> *anyMap = nullptr;
    for (Value v : vecVals) {
      if (dimMap.count(v)) {
        anyMap = &dimMap[v];
        break;
      }
    }
    if (anyMap) {
      SmallVector<int64_t> m = *anyMap;
      for (Value v : vecVals) {
        setMap(v, m);
      }
    }
  }

  // Relaxation helper for `npuv::BroadcastOp`.
  static bool relaxBroadcast(npuv::BroadcastOp bc, DimMap &dimMap, const llvm::DenseSet<Value> &unitVals,
                             const std::function<bool(Value)> &known,
                             const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    if (!known(bc.getResult())) {
      return false;
    }
    SmallVector<int64_t> m = getBroadcastDimMap(bc);
    if (!isNpuVectorType(bc.getSource().getType()) || unitVals.count(bc.getSource())) {
      return false;
    }
    SmallVector<int64_t> &resMap = dimMap[bc.getResult()];
    SmallVector<int64_t> srcMap;
    for (int64_t i = 0; i < static_cast<int64_t>(m.size()); ++i) {
      if (m[i] >= 0 && m[i] < static_cast<int64_t>(resMap.size())) {
        srcMap.push_back(resMap[m[i]]);
      }
    }
    setMap(bc.getSource(), srcMap);
    return true;
  }

  // Relaxation helper for `npuv::TransposeOp`.
  static bool relaxTranspose(npuv::TransposeOp tr, DimMap &dimMap, const std::function<bool(Value)> &known,
                             const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    // result dim i == source dim perm[i]; so result and source maps are a
    // permutation of one another. Derive whichever side is still unknown.
    ArrayRef<int64_t> perm = tr.getPermutation();
    int64_t n = static_cast<int64_t>(perm.size());
    if (known(tr.getVector()) && !known(tr.getResult())) {
      SmallVector<int64_t> &srcMap = dimMap[tr.getVector()];
      if (static_cast<int64_t>(srcMap.size()) == n) {
        SmallVector<int64_t> resMap(n);
        for (int64_t i = 0; i < n; ++i) {
          resMap[i] = srcMap[perm[i]];
        }
        setMap(tr.getResult(), resMap);
      }
    }
    if (known(tr.getResult()) && !known(tr.getVector())) {
      SmallVector<int64_t> &resMap = dimMap[tr.getResult()];
      if (static_cast<int64_t>(resMap.size()) == n) {
        SmallVector<int64_t> srcMap(n);
        for (int64_t i = 0; i < n; ++i) {
          srcMap[perm[i]] = resMap[i];
        }
        setMap(tr.getVector(), srcMap);
      }
    }
    return true;
  }

  // Relaxation helper for `npuv::ReductionOp`.
  // Derive the (rank-reduced) result map from a known source: the result
  // keeps every non-reduced source dim in order. This is the direction used
  // by partial reductions that reduce the register axis and keep an outer
  // axis (the kept axis is not the trailing loop dim).
  static void deriveReducedDestMap(npuv::ReductionOp red, const DimMap &dimMap, ArrayRef<bool> reduced,
                                   int64_t srcRank, const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    SmallVector<int64_t> srcMap = dimMap.find(red.getVector())->second;
    if (static_cast<int64_t>(srcMap.size()) != srcRank) {
      return;
    }
    SmallVector<int64_t> destMap;
    for (int64_t d = 0; d < srcRank; ++d) {
      if (!reduced[d]) {
        destMap.push_back(srcMap[d]);
      }
    }
    setMap(red.getDest(), destMap);
  }

  static bool relaxReduction(npuv::ReductionOp red, const DimMap &dimMap, const std::function<bool(Value)> &known,
                             const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    // Derive source from result + reduced dims (reduced source dim d maps to
    // loop dim d, since the reduction source spans the full grid).
    int64_t srcRank = npuVectorRank(red.getVector().getType());
    SmallVector<bool> reduced(srcRank, false);
    if (auto dims = red.getReductionDims(); dims && !dims->empty()) {
      for (int64_t d : *dims) {
        if (d >= 0 && d < srcRank) {
          reduced[d] = true;
        }
      }
    } else {
      reduced.assign(srcRank, true);
    }
    if (isNpuVectorType(red.getDest().getType()) && known(red.getVector()) && !known(red.getDest())) {
      deriveReducedDestMap(red, dimMap, reduced, srcRank, setMap);
    }

    SmallVector<int64_t> resMap;
    if (isNpuVectorType(red.getDest().getType()) && known(red.getDest())) {
      resMap = dimMap.find(red.getDest())->second;
    } else if (!isNpuVectorType(red.getDest().getType())) {
      resMap = {};  // scalar destination
    } else {
      return false;
    }
    SmallVector<int64_t> srcMap;
    unsigned ri = 0;
    for (int64_t d = 0; d < srcRank; ++d) {
      if (reduced[d]) {
        srcMap.push_back(d);
      } else if (ri < resMap.size()) {
        srcMap.push_back(resMap[ri++]);
      } else {
        srcMap.push_back(d);
      }
    }
    setMap(red.getVector(), srcMap);
    return true;
  }

  // Relaxation helper for `npuv::ExtractSliceOp`. Result dim `r` is projected
  // from source dim `keep_dims[r]`; dropped (rank-reduced) source dims are
  // size-1 and never carry the register axis, so they get the sentinel -1.
  static bool relaxExtractSlice(npuv::ExtractSliceOp es, DimMap &dimMap, const std::function<bool(Value)> &known,
                                const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    ArrayRef<int64_t> keep = es.getKeepDims();
    int64_t srcRank = npuVectorRank(es.getSource().getType());
    if (known(es.getSource()) && !known(es.getResult())) {
      SmallVector<int64_t> srcMap = dimMap.find(es.getSource())->second;
      SmallVector<int64_t> resMap;
      for (int64_t r = 0; r < static_cast<int64_t>(keep.size()); ++r) {
        int64_t sd = keep[r];
        resMap.push_back((sd >= 0 && sd < static_cast<int64_t>(srcMap.size())) ? srcMap[sd] : -1);
      }
      setMap(es.getResult(), resMap);
    }
    if (known(es.getResult()) && !known(es.getSource())) {
      SmallVector<int64_t> resMap = dimMap.find(es.getResult())->second;
      SmallVector<int64_t> srcMap(srcRank, -1);
      for (int64_t r = 0; r < static_cast<int64_t>(keep.size()) && r < static_cast<int64_t>(resMap.size()); ++r) {
        int64_t sd = keep[r];
        if (sd >= 0 && sd < srcRank) {
          srcMap[sd] = resMap[r];
        }
      }
      setMap(es.getSource(), srcMap);
    }
    return true;
  }

  // Relaxation helper for `npuv::InsertSliceOp`. The result type equals the
  // destination type, so they share a map; the inserted source (same rank as
  // the slice) maps onto the same loop dims.
  static bool relaxInsertSlice(npuv::InsertSliceOp is, DimMap &dimMap, const std::function<bool(Value)> &known,
                               const std::function<void(Value, ArrayRef<int64_t>)> &setMap) {
    if (known(is.getDest()) && !known(is.getResult())) {
      SmallVector<int64_t> destMap = dimMap.find(is.getDest())->second;
      setMap(is.getResult(), destMap);
    }
    if (known(is.getResult()) && !known(is.getDest())) {
      SmallVector<int64_t> resMap = dimMap.find(is.getResult())->second;
      setMap(is.getDest(), resMap);
    }
    Value ref;
    if (known(is.getResult())) {
      ref = is.getResult();
    } else if (known(is.getDest())) {
      ref = is.getDest();
    }
    if (ref && !known(is.getSource())) {
      SmallVector<int64_t> refMap = dimMap.find(ref)->second;
      if (npuVectorRank(is.getSource().getType()) == static_cast<int64_t>(refMap.size())) {
        setMap(is.getSource(), refMap);
      }
    }
    return true;
  }

  // True when `v` is a reduction result, or is computed from one only through
  // rank-preserving ops (elementwise arith/math, npuvector casts, cmp/select).
  // The walk stops at broadcast / transpose / transfer_read, which (re)build a
  // full-grid or register-row shape and therefore break the "per-row scalar"
  // chain (a broadcast of a reduced scalar back across the row stays reg-tiled).
  static bool isReductionDerived(Value v) {
    SmallVector<Value> work{v};
    llvm::DenseSet<Value> seen;
    while (!work.empty()) {
      Value cur = work.pop_back_val();
      if (!seen.insert(cur).second) {
        continue;
      }
      Operation *def = cur.getDefiningOp();
      if (!def) {
        continue;
      }
      if (isa<npuv::ReductionOp>(def)) {
        return true;
      }
      if (isa<npuv::BroadcastOp, npuv::TransposeOp, npuv::TransferReadOp>(def)) {
        continue;
      }
      for (Value o : def->getOperands()) {
        if (isNpuVectorType(o.getType())) {
          work.push_back(o);
        }
      }
    }
    return false;
  }

  // Compute dimMap for every non-unit npuvector value in the kernel.
  // `seedAnchor` controls whether the anchor (a transfer value spanning the
  // whole grid) is identity-seeded. It must be false for transpose kernels,
  // where the read side is a permutation of the (identity-seeded) write side
  // and must be derived instead of forced to identity.
  void computeDimMap(Block &entry, Value anchorVal, int64_t loopRank, DimMapState state,
                     bool seedAnchor = true) const {
    DimMap &dimMap = state.dimMap;
    llvm::DenseSet<Value> &unitVals = state.unitVals;
    // 1. Classify unit values.
    for (Operation &op : entry) {
      for (Value v : op.getResults()) {
        if (isUnitNPUVector(v.getType())) {
          unitVals.insert(v);
        }
      }
    }
    // 2. Seed: the anchor spans the full iteration space, and every write value
    //    is identity-aligned to the trailing loop dims.
    auto identityFor = [&loopRank](int64_t rank) {
      SmallVector<int64_t> m;
      for (int64_t d = loopRank - rank; d < loopRank; ++d) {
        m.push_back(d);
      }
      return m;
    };
    if (seedAnchor && anchorVal && !unitVals.count(anchorVal)) {
      dimMap[anchorVal] = identityFor(npuVectorRank(anchorVal.getType()));
    }
    for (Operation &op : entry) {
      if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        Value v = wr.getVector();
        // A reduction result (possibly followed by rank-preserving casts /
        // elementwise ops) is not identity-aligned to the trailing loop dims:
        // it keeps only the non-reduced axes. Leave it unseeded so relaxation
        // derives the correct (kept-dim) map from the reduction source instead.
        // A broadcast re-expands a reduced scalar back to the register row, so
        // isReductionDerived deliberately stops at broadcasts (that write stays
        // reg-tiled and is seeded normally).
        if (isReductionDerived(v)) {
          continue;
        }
        if (isNpuVectorType(v.getType()) && !unitVals.count(v) && !dimMap.count(v)) {
          dimMap[v] = identityFor(npuVectorRank(v.getType()));
        }
      }
    }
    // 3. Relax to a fixpoint (small straight-line kernels converge quickly).
    bool changed = true;
    int64_t guard = 0;
    int64_t maxIter = static_cast<int64_t>(entry.getOperations().size()) + 2;
    while (changed && guard++ <= maxIter) {
      changed = false;
      for (Operation &op : entry) {
        changed |= relaxDimMap(&op, loopRank, dimMap, unitVals);
      }
    }
  }

  // True when one of the value's dims is the register (innermost loop) dim.
  // For elementwise/broadcast that dim is always the trailing one; for a
  // transpose-fed read it can be an interior dim (the read becomes strided),
  // hence the membership test rather than a back() comparison.
  bool isRegTiled(Value v, const DimMap &dimMap, const llvm::DenseSet<Value> &unitVals, int64_t loopRank) const {
    if (unitVals.count(v)) {
      return false;
    }
    auto it = dimMap.find(v);
    if (it == dimMap.end() || it->second.empty()) {
      return false;
    }
    return llvm::is_contained(it->second, loopRank - 1);
  }

  // The value dim that carries the register (innermost loop) axis, or -1.
  static int64_t regTiledDim(ArrayRef<int64_t> m, int64_t loopRank) {
    for (int64_t k = 0; k < static_cast<int64_t>(m.size()); ++k) {
      if (m[k] == loopRank - 1) {
        return k;
      }
    }
    return -1;
  }

  // -------------------------------------------------------------------------
  // Per-op emission inside a (already-built) loop body.
  // -------------------------------------------------------------------------

  struct TileCtx {
    int64_t loopRank = 0;
    int64_t laneCount = 1;
    ArrayRef<Value> ivs;  // one induction var per loop dim; empty for hoisted units
    // Optional register mask for the innermost dim. Left null in the aligned
    // (full-tile) main loops; only the masked remainder/tail loops set it.
    Value mask;
    const DimMap *dimMap = nullptr;
    const llvm::DenseSet<Value> *unitVals = nullptr;
  };

  // Reduction-related context shared across partial-reduction pass helpers.
  struct ReductionCtx {
    ArrayRef<npuv::ReductionOp> reductions;
    ArrayRef<Value> redIdents;
    ArrayRef<vector::CombiningKind> redKinds;
    ArrayRef<Operation *> unitEmitted;
    const llvm::DenseSet<Operation *> *dependsOnReduction = nullptr;
  };

  // Partial-pass environment shared across pass helpers.
  struct PartialPassEnv {
    OpBuilder &builder;
    Block &entry;
    IRMapping &shared;
    const TileCtx &baseCtx;
  };

  // Loop iteration context for partial-pass runners.
  struct LoopIterCtx {
    ArrayRef<Value> bounds;
    ArrayRef<Value> outerIvs;
    Value c0;
    Value cLane;
    int64_t rank = 0;
  };

  // Full-reduction specification shared by buildReductionRegLoop / buildFullReductionLoopNest.
  struct FullReductionInfo {
    npuv::ReductionOp redOp;
    vector::CombiningKind kind;
    Type scalarElem;
    ArrayRef<Operation *> producers;
    Value regIdent;
    VectorType regVecTy;
    int64_t laneCount = 1;
  };

  // Register-loop parameters shared across loop emitters.
  struct PassLoopParams {
    Value lbv;
    Value ubv;
    Value bound;
    Value cLane;
    ArrayRef<Value> outerIvs;
    ArrayRef<Value> initAccs;
    bool masked = false;
  };

  // Loop constants c0 / c1 / cLane.
  struct LoopConstants {
    Value c0;
    Value c1;
    Value cLane;
  };

  // Reduced value + its scalar element type.
  struct ReducedValue {
    Value reduced;
    Type scalarElem;
  };

  // Loop nest dimensions: rank + bounds.
  struct LoopNestDims {
    int64_t rank = 0;
    ArrayRef<Value> bounds;
  };

  int64_t widthOf(Value v, const TileCtx &ctx) const {
    if (ctx.unitVals->count(v)) {
      return 1;
    }
    auto it = ctx.dimMap->find(v);
    if (it == ctx.dimMap->end() || it->second.empty()) {
      return 1;
    }
    return llvm::is_contained(it->second, ctx.loopRank - 1) ? ctx.laneCount : 1;
  }

  // Permutation map for a transfer whose value has dimMap `m`. The 1-D register
  // vector maps to the memref dim that carries the register (innermost loop)
  // axis. For elementwise/broadcast transfers that is the trailing dim, yielding
  // the usual minor-identity map; for a transpose-fed read it is an interior dim
  // and the transfer becomes a strided (gather/scatter) access -- still a single
  // register-sized vector, so the register-size invariant holds.
  // Special case: when the source memref is rank 0 (npuvector+scalar form), the
  // 1-D vector cannot reference any source dim, so the map collapses to a
  // broadcast (`() -> (0)`) -- the single source element is replicated across
  // every lane.
  AffineMap transferPermMap(OpBuilder &b, unsigned memRank, ArrayRef<int64_t> m, int64_t w, const TileCtx &ctx) const {
    if (memRank == 0) {
      return getRank0BroadcastPermMap(b, 1);
    }
    if (w == ctx.laneCount) {
      int64_t vecDim = regTiledDim(m, ctx.loopRank);
      if (vecDim >= 0 && vecDim < static_cast<int64_t>(memRank)) {
        return AffineMap::get(memRank, 0, b.getAffineDimExpr(static_cast<unsigned>(vecDim)), b.getContext());
      }
    }
    return AffineMap::getMinorIdentityMap(memRank, 1, b.getContext());
  }

  // Build the per-memref-dim indices for a transfer whose value has dimMap `m`.
  SmallVector<Value> buildTransferIndices(BuilderEnv env, ValueRange origIndices, ArrayRef<int64_t> m,
                                          IRMapping &map, const TileCtx &ctx) const {
    SmallVector<Value> indices;
    for (size_t k = 0; k < origIndices.size(); ++k) {
      Value base = map.lookupOrDefault(origIndices[k]);
      Value iv;
      if (!ctx.ivs.empty() && k < m.size() && m[k] >= 0 && m[k] < static_cast<int64_t>(ctx.ivs.size())) {
        iv = ctx.ivs[m[k]];
      }
      indices.push_back(addIndex(env.builder, env.loc, base, iv));
    }
    return indices;
  }

  // Lower one op into its register-sized vector form, recording the result(s)
  // in `map`. Handles transfer_read/write, broadcast (dimension-aware),
  // elementwise, casts, cmp, select and constants. Reduction/transpose are
  // handled by their dedicated tilers.
  LogicalResult emitTiledOp(OpBuilder &b, Operation *op, IRMapping &map, const TileCtx &ctx) const {
    Location loc = op->getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };

    if (auto rd = dyn_cast<npuv::TransferReadOp>(op)) {
      return emitTiledTransferRead(b, rd, map, ctx);
    }
    if (auto wr = dyn_cast<npuv::TransferWriteOp>(op)) {
      return emitTiledTransferWrite(b, wr, map, ctx);
    }
    if (auto bc = dyn_cast<npuv::BroadcastOp>(op)) {
      return emitTiledBroadcast(b, bc, map, ctx);
    }
    if (auto tr = dyn_cast<npuv::TransposeOp>(op)) {
      // The transpose is fully realized by the (permuted) dim maps of its
      // producers/consumers: the source and result share the same register-lane
      // data, so the register tile passes straight through. The reordering of
      // the non-register dims is carried by the loop indices that surrounding
      // transfers build from their own dim maps.
      map.map(tr.getResult(), remap(tr.getVector()));
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
    if (auto es = dyn_cast<npuv::ExtractSliceOp>(op)) {
      return emitTiledExtractSlice(b, es, map, ctx);
    }
    if (auto is = dyn_cast<npuv::InsertSliceOp>(op)) {
      return emitTiledInsertSlice(b, is, map, ctx);
    }
    if (succeeded(emitTiledUnaryCast(b, op, map, ctx))) {
      return success();
    }
    if (llvm::isa_and_nonnull<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
      return emitTiledArithMath(b, op, map, ctx);
    }
    return op->emitError("npuvector-to-vector: unsupported op inside tileable kernel");
  }

  // `emitTiledOp` helper: lower a `npuv::TransferReadOp`.
  LogicalResult emitTiledTransferRead(OpBuilder &b, npuv::TransferReadOp rd, IRMapping &map, const TileCtx &ctx) const {
    Location loc = rd.getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    auto vt = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType());
    Type elemType = vt.getElementType();
    int64_t w = widthOf(rd.getResult(), ctx);
    ArrayRef<int64_t> m = ctx.dimMap->count(rd.getResult())
                            ? ArrayRef<int64_t>(ctx.dimMap->find(rd.getResult())->second)
                            : ArrayRef<int64_t>();
    SmallVector<Value> indices = buildTransferIndices({b, loc}, rd.getIndices(), m, map, ctx);
    int64_t srcRank = getShapedSourceRank(rd.getSource());
    if (srcRank >= 0 && static_cast<int64_t>(indices.size()) > srcRank) {
      indices.erase(indices.begin(), indices.begin() + (indices.size() - srcRank));
    }
    auto memRank = static_cast<unsigned>(indices.size());
    auto flatTy = VectorType::get({w}, elemType);

    // When the register axis falls on an interior memref dim (the transpose
    // case), transferPermMap produces a 1-D projection map like
    // (d0,d1)->(d0). That map is not a permutation, so the downstream
    // transfer_read→gather lowering (which requires permMap.isPermutation())
    // never fires.
    int64_t vecDim = (w == ctx.laneCount) ? regTiledDim(m, ctx.loopRank) : -1;
    bool needPermute = (vecDim >= 0 && vecDim < static_cast<int64_t>(memRank) &&
                        static_cast<unsigned>(vecDim) != memRank - 1 && memRank > 1);

    Value res;
    if (needPermute) {
      SmallVector<int64_t> vecShape(memRank, 1);
      vecShape.back() = w;
      auto multiTy = VectorType::get(vecShape, elemType);

      SmallVector<AffineExpr> exprs(memRank);
      for (unsigned d = 0; d < memRank; ++d) {
        exprs[d] = b.getAffineDimExpr(d);
      }
      std::swap(exprs[vecDim], exprs[memRank - 1]);
      AffineMap permMap = AffineMap::get(memRank, 0, exprs, b.getContext());

      res = b.create<vector::TransferReadOp>(loc, multiTy, remap(rd.getSource()), indices, permMap,
                                             remap(rd.getPadding()), Value(),
                                             b.getBoolArrayAttr(SmallVector<bool>(memRank, true)));
      res = b.create<vector::ShapeCastOp>(loc, flatTy, res);
    } else {
      AffineMap permMap = transferPermMap(b, memRank, m, w, ctx);
      // Aligned main loops carry no mask (ctx.mask is null); only the masked
      // tail loop restricts the active lanes. Rank-0 sources can never overflow
      // their single element, so they stay unmasked regardless.
      Value mask = (memRank > 0 && w == ctx.laneCount) ? ctx.mask : Value();
      res = b.create<vector::TransferReadOp>(loc, flatTy, remap(rd.getSource()), indices, permMap,
                                             remap(rd.getPadding()), mask, b.getBoolArrayAttr({true}));
    }
    map.map(rd.getResult(), res);
    return success();
  }

  // `emitTiledOp` helper: lower a `npuv::TransferWriteOp`.
  LogicalResult emitTiledTransferWrite(OpBuilder &b, npuv::TransferWriteOp wr,
                                       IRMapping &map, const TileCtx &ctx) const {
    Location loc = wr.getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    Value val = wr.getVector();
    int64_t w = widthOf(val, ctx);
    ArrayRef<int64_t> m =
      ctx.dimMap->count(val) ? ArrayRef<int64_t>(ctx.dimMap->find(val)->second) : ArrayRef<int64_t>();
    SmallVector<Value> indices = buildTransferIndices({b, loc}, wr.getIndices(), m, map, ctx);
    int64_t srcRank = getShapedSourceRank(wr.getSource());
    if (srcRank >= 0 && static_cast<int64_t>(indices.size()) > srcRank) {
      indices.erase(indices.begin(), indices.begin() + (indices.size() - srcRank));
    }

    Value vecVal = remap(val);
    if (srcRank == 0) {
      emitRank0TransferWrite(b, loc, vecVal, remap(wr.getSource()));
      return success();
    }
    auto memRank = static_cast<unsigned>(indices.size());
    AffineMap permMap = transferPermMap(b, memRank, m, w, ctx);
    // Aligned main loops carry no mask (ctx.mask is null); only the masked tail
    // loop restricts the active lanes written back.
    Value mask = (w == ctx.laneCount) ? ctx.mask : Value();
    b.create<vector::TransferWriteOp>(loc, Type(), vecVal, remap(wr.getSource()), indices, permMap,
                                      mask, b.getBoolArrayAttr({true}));
    return success();
  }

  // `emitTiledOp` helper: lower a `npuv::BroadcastOp` (dimension-aware).
  LogicalResult emitTiledBroadcast(OpBuilder &b, npuv::BroadcastOp bc, IRMapping &map, const TileCtx &ctx) const {
    Location loc = bc.getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    Type elemType = llvm::cast<npuv::NPUVectorType>(bc.getResult().getType()).getElementType();
    int64_t w = widthOf(bc.getResult(), ctx);
    Value src = bc.getSource();
    Value srcVal = remap(src);
    bool srcRegTiled = isNpuVectorType(src.getType()) && widthOf(src, ctx) == ctx.laneCount;
    // When both the source and result carry the register axis, the broadcast
    // is the identity along that axis (the extra broadcast dims are handled by
    // the surrounding loop): pass the source register tile straight through.
    if (w == ctx.laneCount && srcRegTiled) {
      map.map(bc.getResult(), srcVal);
      return success();
    }
    // Otherwise the result's register axis is a *new* broadcast dim: splat the
    // (scalar / unit) source value across all lanes.
    auto vecTy = VectorType::get({w}, elemType);
    Value res = b.create<vector::BroadcastOp>(loc, vecTy, srcVal);
    map.map(bc.getResult(), res);
    return success();
  }

  // `emitTiledOp` helper: lower a `npuv::ExtractSliceOp`.
  // Inside the register-tiled loop the slice degenerates to a passthrough of
  // the (already register-sized) source tile: the loop bounds were derived from
  // the slice sizes, so the induction variables already walk exactly the slice
  // region. This is only valid when every offset is zero (unit strides); a
  // non-zero offset would shift the loaded region and is rejected so we never
  // silently miscompile.
  LogicalResult emitTiledExtractSlice(OpBuilder &b, npuv::ExtractSliceOp es, IRMapping &map, const TileCtx &ctx) const {
    if (!allZeroOffsets(es.getOffsets()) || !allUnitStrides(es.getStrides())) {
      return es.emitError("npuvector-to-vector: only zero-offset unit-stride extract_slice is supported in a "
                          "tileable kernel");
    }
    map.map(es.getResult(), map.lookupOrDefault(es.getSource()));
    return success();
  }

  // `emitTiledOp` helper: lower a `npuv::InsertSliceOp`. Same reasoning as
  // extract_slice: the surrounding transfer_write walks exactly the slice
  // region, so the register tile of the inserted source passes straight through.
  LogicalResult emitTiledInsertSlice(OpBuilder &b, npuv::InsertSliceOp is, IRMapping &map, const TileCtx &ctx) const {
    if (!allZeroOffsets(is.getOffsets()) || !allUnitStrides(is.getStrides())) {
      return is.emitError("npuvector-to-vector: only zero-offset unit-stride insert_slice is supported in a "
                          "tileable kernel");
    }
    map.map(is.getResult(), map.lookupOrDefault(is.getSource()));
    return success();
  }

  static bool allZeroOffsets(ValueRange offsets) {
    return llvm::all_of(offsets, [](Value v) { return isConstantZeroIndex(v); });
  }

  static bool allUnitStrides(ValueRange strides) {
    return llvm::all_of(strides, [](Value v) {
      auto c = getConstantIndex(v);
      return c && *c == 1;
    });
  }

  LogicalResult emitTiledUnaryCast(OpBuilder &b, Operation *op, IRMapping &map, const TileCtx &ctx) const {
    if (!isUnaryNpuvCast(op)) {
      return failure();
    }
    Location loc = op->getLoc();
    int64_t w = widthOf(op->getResult(0), ctx);
    Type outElem = llvm::cast<npuv::NPUVectorType>(op->getResult(0).getType()).getElementType();
    Type outTy = VectorType::get({w}, outElem);
    Value res = emitCastFromNpuv(b, loc, op, outTy, map.lookupOrDefault(op->getOperand(0)));
    if (!res) {
      return failure();
    }
    map.map(op->getResult(0), res);
    return success();
  }

  LogicalResult emitTiledArithMath(OpBuilder &b, Operation *op, IRMapping &map, const TileCtx &ctx) const {
    Location loc = op->getLoc();
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(cst.getType());
      if (!nvt) {
        Operation *cloned = b.clone(*op);
        map.map(cst.getResult(), cloned->getResult(0));
        return success();
      }
      int64_t w = widthOf(cst.getResult(), ctx);
      auto vecType = VectorType::get({w}, nvt.getElementType());
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat()) {
        return op->emitError("npuvector-to-vector: only splat npuvector constants are supported");
      }
      Value res = b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vecType, dense.getSplatValue<Attribute>()));
      map.map(cst.getResult(), res);
      return success();
    }

    SmallVector<Type> resultTypes;
    for (Value r : op->getResults()) {
      if (auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(r.getType())) {
        resultTypes.push_back(VectorType::get({widthOf(r, ctx)}, nvt.getElementType()));
      } else {
        resultTypes.push_back(r.getType());
      }
    }
    SmallVector<Value> operands(op->getNumOperands());
    std::transform(op->operand_begin(), op->operand_end(), operands.begin(),
                   [&](Value v) { return map.lookupOrDefault(v); });
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

  // Emit all unit (loop-invariant) ops once, before the loop nest.
  LogicalResult emitUnitOps(OpBuilder &b, Block &entry, IRMapping &map, const TileCtx &ctx,
                            SmallVectorImpl<Operation *> &emitted) const {
    for (Operation &op : entry.without_terminator()) {
      bool isUnit = !op.getResults().empty() && llvm::all_of(op.getResults(), [&](Value v) {
        return ctx.unitVals->count(v) || !isNpuVectorType(v.getType());
      }) && touchesNpuVector(&op);
      if (!isUnit) {
        continue;
      }
      if (failed(emitTiledOp(b, &op, map, ctx))) {
        return failure();
      }
      emitted.push_back(&op);
    }
    return success();
  }

  // -------------------------------------------------------------------------
  // Anchor selection: the highest-rank npuvector transfer value, which spans
  // the full iteration space (output for broadcast kernels, input otherwise).
  // -------------------------------------------------------------------------
  static Value findAnchorValue(Block &entry) {
    Value anchor;
    int64_t bestRank = -1;
    auto consider = [&bestRank, &anchor](Value v) {
      if (!isNpuVectorType(v.getType()) || isUnitNPUVector(v.getType())) {
        return;
      }
      int64_t r = npuVectorRank(v.getType());
      if (r > bestRank) {
        bestRank = r;
        anchor = v;
      }
    };
    for (Operation &op : entry) {
      if (auto rd = dyn_cast<npuv::TransferReadOp>(&op)) {
        consider(rd.getResult());
      } else if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        consider(wr.getVector());
      }
    }
    return anchor;
  }

  // Drop the npuvector compute ops (and any now-dead constants) once they have
  // been replaced by the tiled loop.
  static void eraseComputeOps(Block &entry, ArrayRef<Operation *> computeOps) {
    for (auto it = computeOps.rbegin(); it != computeOps.rend(); ++it) {
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
  }

  // =========================================================================
  // Shared register-tiled loop-nest emitter.
  // Builds the outer loop nest (one step-1 loop per non-register dim) and splits
  // the innermost (register) dim into an aligned main loop and a masked tail
  // loop via `planRegSplit`. The tail loop masks its lanes to the (dynamic or
  // static) remainder so out-of-range lanes are never read, computed on, or
  // written back -- this is what keeps a kernel whose live extent is a dynamic
  // (e.g. `affine.min`) or otherwise unaligned length from pulling in padding
  // garbage. For a statically lane-aligned extent the tail is elided and the
  // main loop lowers byte-identically to the historical unmasked form.
  // Each inner loop gets a fresh op map seeded with the loop-invariant unit
  // values so the two halves never share (register-specific) mappings.
  // =========================================================================
  LogicalResult emitSplitTiledLoopNest(OpBuilder &builder, Location loc, ArrayRef<Value> bounds,
                                       ArrayRef<Operation *> computeOps, ArrayRef<Operation *> unitEmitted,
                                       IRMapping &unitMap, const TileCtx &unitCtx) const {
    int64_t R = unitCtx.loopRank;
    int64_t lanes = unitCtx.laneCount;
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cLane = builder.create<arith::ConstantIndexOp>(loc, lanes);

    // Outer loop nest over the non-register dims (step 1).
    SmallVector<Value> outerIvs;
    OpBuilder body = builder;
    for (int64_t d = 0; d < R - 1; ++d) {
      auto forOp = body.create<scf::ForOp>(loc, c0, bounds[d], c1);
      outerIvs.push_back(forOp.getInductionVar());
      body = OpBuilder::atBlockBegin(forOp.getBody());
    }

    Value bound = bounds[R - 1];
    auto emitInner = [&](OpBuilder &ib, Value lbv, Value ubv, bool masked) -> LogicalResult {
      auto innerFor = ib.create<scf::ForOp>(loc, lbv, ubv, cLane);
      Value regIv = innerFor.getInductionVar();
      OpBuilder lb = OpBuilder::atBlockBegin(innerFor.getBody());

      SmallVector<Value> ivs(outerIvs.begin(), outerIvs.end());
      ivs.push_back(regIv);

      // Fresh op map per inner loop, seeded with the loop-invariant unit values.
      IRMapping innerMap;
      for (Operation *u : unitEmitted) {
        for (Value r : u->getResults()) {
          if (Value mapped = unitMap.lookupOrDefault(r); mapped != r) {
            innerMap.map(r, mapped);
          }
        }
      }
      TileCtx ctx = unitCtx;
      ctx.ivs = ivs;
      if (masked) {
        ctx.mask = tailRegMask(lb, loc, bound, regIv, lanes);
      }
      for (Operation *op : computeOps) {
        if (llvm::is_contained(unitEmitted, op)) {
          continue;
        }
        if (failed(emitTiledOp(lb, op, innerMap, ctx))) {
          return failure();
        }
      }
      return success();
    };

    RegSplit split = planRegSplit(body, loc, bound, lanes);
    if (split.emitMain) {
      if (failed(emitInner(body, c0, split.alignedBound, false))) {
        return failure();
      }
    }
    if (split.emitTail) {
      Value tailLb = split.emitMain ? split.alignedBound : c0;
      if (failed(emitInner(body, tailLb, bound, true))) {
        return failure();
      }
    }
    return success();
  }

  // =========================================================================
  // Elementwise / broadcast kernel tiler
  // =========================================================================
  LogicalResult tileElementwiseKernel(func::FuncOp vfFunc, const KernelPlan &plan) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();
    int64_t lanes = plan.laneCount;

    Value anchor = findAnchorValue(entry);
    if (!anchor) {
      return failure();
    }
    int64_t R = plan.loopRank;
    if (R == 0) {
      return failure();
    }

    DimMap dimMap;
    llvm::DenseSet<Value> unitVals;
    computeDimMap(entry, anchor, R, {dimMap, unitVals});

    OpBuilder builder(entry.getTerminator());
    // Prefer the aggregated loop bounds, which honor extract_slice sizes: when a
    // slice narrows the live range to a dynamic (e.g. `affine.min`) length, that
    // dynamic extent -- not the full padded transfer_read extent -- is what the
    // loop (and its masked tail) must walk. Fall back to the anchor's per-dim
    // extents when the dim map does not fully resolve.
    SmallVector<Value> bounds;
    if (failed(getLoopBounds({builder, loc}, entry, dimMap, R, bounds))) {
      bounds.clear();
      if (failed(getPerDimBounds(builder, loc, anchor, bounds)) || static_cast<int64_t>(bounds.size()) != R) {
        return failure();
      }
    }

    // Collect compute ops, emit unit ops once before the loop.
    SmallVector<Operation *> computeOps;
    for (Operation &op : entry.without_terminator()) {
      if (touchesNpuVector(&op)) {
        computeOps.push_back(&op);
      }
    }
    if (computeOps.empty()) {
      return success();
    }

    IRMapping map;
    TileCtx unitCtx;
    unitCtx.loopRank = R;
    unitCtx.laneCount = lanes;
    unitCtx.dimMap = &dimMap;
    unitCtx.unitVals = &unitVals;
    SmallVector<Operation *> unitEmitted;
    if (failed(emitUnitOps(builder, entry, map, unitCtx, unitEmitted))) {
      return failure();
    }

    // Build the loop nest with an aligned main loop and a masked tail so an
    // unaligned / dynamic register extent never reads or writes past its live
    // range (which would corrupt e.g. a reduction accumulated across the tile).
    if (failed(emitSplitTiledLoopNest(builder, loc, bounds, computeOps, unitEmitted, map, unitCtx))) {
      return failure();
    }

    eraseComputeOps(entry, computeOps);
    return success();
  }

  // =========================================================================
  // Transpose kernel tiler
  // The transpose is treated as a permutation of the per-value dim maps rather
  // than a data-shuffling op: the register-lane axis is shared by the transpose
  // source and result, so the register tile passes straight through while the
  // reordering of the remaining axes is carried by the loop indices that each
  // transfer builds from its own (permuted) dim map.
  // This reuses the general per-op emission engine, so it transparently covers:
  //   * any number of transposes in one vf;
  //   * arbitrary permutations, including inner-axis transposes (the producing
  //     read simply becomes a strided/gather transfer of a register-sized
  //     vector -- never an oversized vector.transpose);
  //   * transposes fused with surrounding elementwise / broadcast / cast ops.
  // No oversized vector is ever materialized, so the register-size invariant
  // (and hence the no-stack-overflow guarantee) always holds.
  // =========================================================================
  LogicalResult tileTransposeKernel(func::FuncOp vfFunc, const KernelPlan &plan) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();
    int64_t lanes = plan.laneCount;
    int64_t R = plan.loopRank;
    if (R == 0) {
      return failure();
    }

    // Dim maps: seed from the (identity-aligned) writes and derive the read /
    // transpose sides; do NOT identity-seed the read anchor, since under a
    // transpose it is a permutation of the write side.
    Value anchor = findAnchorValue(entry);
    DimMap dimMap;
    llvm::DenseSet<Value> unitVals;
    computeDimMap(entry, anchor, R, {dimMap, unitVals}, false);

    OpBuilder builder(entry.getTerminator());
    SmallVector<Value> bounds;
    if (failed(getLoopBounds({builder, loc}, entry, dimMap, R, bounds))) {
      return failure();
    }

    SmallVector<Operation *> computeOps;
    for (Operation &op : entry.without_terminator()) {
      if (touchesNpuVector(&op)) {
        computeOps.push_back(&op);
      }
    }
    if (computeOps.empty()) {
      return success();
    }

    IRMapping map;
    TileCtx unitCtx;
    unitCtx.loopRank = R;
    unitCtx.laneCount = lanes;
    unitCtx.dimMap = &dimMap;
    unitCtx.unitVals = &unitVals;
    SmallVector<Operation *> unitEmitted;
    if (failed(emitUnitOps(builder, entry, map, unitCtx, unitEmitted))) {
      return failure();
    }

    // Build the outer loop nest over the output index space (step 1). The
    // innermost (register) dim is split into an aligned main loop and a masked
    // tail loop so unaligned extents (e.g. a reduction K-tile whose length is
    // not a multiple of the lane count) never read/write past their live range.
    if (failed(emitSplitTiledLoopNest(builder, loc, bounds, computeOps, unitEmitted, map, unitCtx))) {
      return failure();
    }

    eraseComputeOps(entry, computeOps);
    return success();
  }

  // =========================================================================
  // Full reduction kernel tiler (unchanged shape -- kept for NormalizeVector
  // compatibility and the existing tests).
  // =========================================================================
  static bool scanFullReductionCandidate(func::FuncOp vfFunc, npuv::ReductionOp &redOp) {
    int redCount = 0;
    bool ok = true;
    bool hasRead = false;
    vfFunc.walk([&](Operation *op) {
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

  // True for the unary `!npuvector`-typed cast ops that may sit between a full
  // reduction and the final memref delivery in the upstream npuvector form.
  static bool isUnaryNpuvCast(Operation *op) {
    return isa<npuv::ExtFOp, npuv::TruncFOp, npuv::ExtSIOp, npuv::ExtUIOp, npuv::TruncIOp, npuv::SIToFPOp,
               npuv::UIToFPOp, npuv::FPToSIOp, npuv::FPToUIOp, npuv::BitcastOp, npuv::IndexCastOp, npuv::IndexCastUIOp>(
      op);
  }

  // Reduction delivery chain: from `npuvector.reduction`'s result down through
  // (optional) unary `!npuvector` cast ops to a final store.
  struct ReductionDelivery {
    SmallVector<Operation *> castChain;  // unary npuv casts, in producer->sink order
    Operation *finalWrite = nullptr;     // memref::StoreOp OR npuv::TransferWriteOp
  };

  static bool traceReductionDelivery(npuv::ReductionOp redOp, ReductionDelivery &delivery) {
    Value cur = redOp.getResult();
    // Bound the walk to a reasonable depth so a malformed chain cannot loop.
    constexpr int kMaxChain = 32;
    for (int step = 0; step < kMaxChain; ++step) {
      if (!cur.hasOneUse()) {
        return false;
      }
      Operation *user = *cur.getUsers().begin();
      if (isa<memref::StoreOp>(user) || isa<npuv::TransferWriteOp>(user)) {
        delivery.finalWrite = user;
        return true;
      }
      if (!isUnaryNpuvCast(user) || user->getNumResults() != 1) {
        return false;
      }
      delivery.castChain.push_back(user);
      cur = user->getResult(0);
    }
    return false;
  }

  static bool isFullReductionTileable(func::FuncOp vfFunc) {
    if (vfFunc.getBody().empty()) {
      return false;
    }
    npuv::ReductionOp redOp;
    if (!scanFullReductionCandidate(vfFunc, redOp)) {
      return false;
    }
    if (!isFullReductionShape(redOp)) {
      return false;
    }

    ReductionDelivery delivery;
    if (!traceReductionDelivery(redOp, delivery)) {
      return false;
    }

    Block &entry = vfFunc.getBody().front();
    llvm::DenseSet<Operation *> allowedAfter;
    for (Operation *o : delivery.castChain) {
      allowedAfter.insert(o);
    }
    allowedAfter.insert(delivery.finalWrite);
    bool afterReduction = false;
    for (Operation &op : entry.without_terminator()) {
      if (&op == redOp.getOperation()) {
        afterReduction = true;
        continue;
      }
      if (afterReduction && touchesNpuVector(&op) && !allowedAfter.count(&op)) {
        return false;
      }
    }
    return true;
  }

  // `isFullReductionTileable` helper: validate that the reduction reduces every
  // source dim and that the dest is a scalar / rank-0 npuvector.
  static bool isFullReductionShape(npuv::ReductionOp redOp) {
    // Full reduction = every source dim is reduced. The old form leaves
    // `reduction_dims` absent (or empty) -- meaning "reduce all dims" by the
    // op's contract. The new form lists every source dim explicitly.
    int64_t srcRank = npuVectorRank(redOp.getVector().getType());
    if (auto dims = redOp.getReductionDims(); dims && !dims->empty()) {
      if (static_cast<int64_t>(dims->size()) != srcRank) {
        return false;  // partial reduction goes through the partial-reduction path
      }
      llvm::SmallDenseSet<int64_t, kSmallDenseSetSize> seen;
      for (int64_t d : *dims) {
        if (d < 0 || d >= srcRank) {
          return false;
        }
        seen.insert(d);
      }
      if (static_cast<int64_t>(seen.size()) != srcRank) {
        return false;
      }
    }
    // The dest is either a pure scalar (old form) or a rank-0 `!npuvector`
    // (new "npuvector+scalar" form). Anything else is a partial reduction.
    Type destTy = redOp.getDest().getType();
    if (auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(destTy)) {
      if (nvt.getRank() != 0) {
        return false;
      }
    } else if (!destTy.isIntOrIndexOrFloat()) {
      return false;
    }
    return true;
  }

  static LogicalResult collectTileBounds(BuilderEnv env, npuv::NPUVectorType type, ValueRange dynSizes,
                                         Value source, SmallVectorImpl<Value> &bounds) {
    ArrayRef<int64_t> shape = type.getShape();
    unsigned dynIdx = 0;
    for (int64_t d = 0, rank = type.getRank(); d < rank; ++d) {
      if (!ShapedType::isDynamic(shape[d])) {
        bounds.push_back(env.builder.create<arith::ConstantIndexOp>(env.loc, shape[d]));
      } else if (dynIdx < dynSizes.size()) {
        bounds.push_back(dynSizes[dynIdx++]);
      } else if (source && llvm::isa<MemRefType>(source.getType())) {
        bounds.push_back(env.builder.create<memref::DimOp>(env.loc, source, d));
      } else {
        return failure();
      }
    }
    return success();
  }

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

  // Resolve the actual output memref the delivery chain writes into, along
  // with its element type and rank.
  struct DeliveryTarget {
    Value memref;
    Type elemType;
    int64_t rank = 0;
    bool isStore = false;  // true for memref::StoreOp, false for npuv::TransferWriteOp
  };

  // Delivery context for buildFullReductionSeed.
  struct DeliveryCtx {
    TypedAttr idAttr;
    const DeliveryTarget &target;
    const ReductionDelivery &delivery;
    Value c0;
  };

  // Output bundle for buildFullReductionSeed.
  struct SeedResult {
    bool reuseTargetAsAccBuf = false;
    Value accBuf;
    Value idScalar;
    Value seedVec;
  };

  static DeliveryTarget resolveDeliveryTarget(const ReductionDelivery &delivery) {
    DeliveryTarget out;
    if (auto store = dyn_cast<memref::StoreOp>(delivery.finalWrite)) {
      out.memref = store.getMemRef();
      out.isStore = true;
    } else if (auto wr = dyn_cast<npuv::TransferWriteOp>(delivery.finalWrite)) {
      out.memref = wr.getSource();
      out.isStore = false;
    }
    if (out.memref) {
      auto mt = llvm::cast<MemRefType>(out.memref.getType());
      out.elemType = mt.getElementType();
      out.rank = mt.getRank();
    }
    return out;
  }

  // Lower a single npuvector unary cast op into the matching `arith` cast,
  // producing a result of `outTy` (scalar element type for the scalar form,
  // vector type for the vector form).
  static Value emitCastFromNpuv(OpBuilder &b, Location loc, Operation *castOp, Type outTy, Value input) {
    return llvm::TypeSwitch<Operation *, Value>(castOp)
        .Case<npuv::ExtFOp>([&](auto) { return b.create<arith::ExtFOp>(loc, outTy, input); })
        .Case<npuv::TruncFOp>([&](auto) { return b.create<arith::TruncFOp>(loc, outTy, input); })
        .Case<npuv::ExtSIOp>([&](auto) { return b.create<arith::ExtSIOp>(loc, outTy, input); })
        .Case<npuv::ExtUIOp>([&](auto) { return b.create<arith::ExtUIOp>(loc, outTy, input); })
        .Case<npuv::TruncIOp>([&](auto) { return b.create<arith::TruncIOp>(loc, outTy, input); })
        .Case<npuv::SIToFPOp>([&](auto) { return b.create<arith::SIToFPOp>(loc, outTy, input); })
        .Case<npuv::UIToFPOp>([&](auto) { return b.create<arith::UIToFPOp>(loc, outTy, input); })
        .Case<npuv::FPToSIOp>([&](auto) { return b.create<arith::FPToSIOp>(loc, outTy, input); })
        .Case<npuv::FPToUIOp>([&](auto) { return b.create<arith::FPToUIOp>(loc, outTy, input); })
        .Case<npuv::BitcastOp>([&](auto) { return b.create<arith::BitcastOp>(loc, outTy, input); })
        .Case<npuv::IndexCastOp>([&](auto) { return b.create<arith::IndexCastOp>(loc, outTy, input); })
        .Case<npuv::IndexCastUIOp>([&](auto) { return b.create<arith::IndexCastUIOp>(loc, outTy, input); })
        .Default(Value());
  }

  // Lower a single npuvector unary cast op into the matching scalar arith op
  // (operating on a scalar input).
  static Value emitScalarCastFromNpuv(OpBuilder &b, Location loc, Operation *castOp, Value input) {
    Type outElem = getElementTypeOrSelf(castOp->getResult(0).getType());
    return emitCastFromNpuv(b, loc, castOp, outElem, input);
  }

  // Vector-typed counterpart to `emitScalarCastFromNpuv`: lowers an npuvector
  // unary cast to the matching `arith` cast on a community vector of `outTy`.
  static Value emitVectorCastFromNpuv(OpBuilder &b, Location loc, Operation *castOp, Type outTy, Value input) {
    return emitCastFromNpuv(b, loc, castOp, outTy, input);
  }

  // Tile a single producer op of the full-reduction kernel into the loop body.
  // This mirrors the elementwise emission but uses a trailing-dim alignment for
  // the indices (producers feed one full-grid reduction).
  LogicalResult tileReductionProducer(OpBuilder &b, Operation *op, IRMapping &map, ValueRange indices, Value mask,
                                      int64_t laneCount) const {
    Location loc = op->getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    auto vecTypeOf = [laneCount](Type elemType) { return VectorType::get({laneCount}, elemType); };
    auto sourceRank = [&indices](Value src) -> int64_t {
      auto mt = llvm::dyn_cast<MemRefType>(src.getType());
      return mt ? mt.getRank() : static_cast<int64_t>(indices.size());
    };
    auto trailingIndices = [&indices](int64_t srcRank) {
      SmallVector<Value> result(indices.begin(), indices.end());
      int64_t n = static_cast<int64_t>(result.size());
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
      Type elemType = nvt.getElementType();
      if (isUnitNPUVector(nvt)) {
        SmallVector<int64_t> shape(nvt.getShape().begin(), nvt.getShape().end());
        auto unitVecTy = VectorType::get(shape, elemType);
        SmallVector<Value> unitIndices(rd.getIndices().size());
        std::transform(rd.getIndices().begin(), rd.getIndices().end(), unitIndices.begin(),
                       [&](Value idx) { return remap(idx); });
        // Discard any leading placeholder indices that overshoot the actual
        // source memref rank (the npuvector+scalar form can attach a `[%c0]`
        // even to a rank-0 memref).
        int64_t srcRank = sourceRank(rd.getSource());
        if (static_cast<int64_t>(unitIndices.size()) > srcRank) {
          unitIndices.erase(unitIndices.begin(), unitIndices.begin() + (unitIndices.size() - srcRank));
        }
        // Rank-0 source feeding a rank>0 unit vector: broadcast permutation
        // map (`() -> (0, ..., 0)`); otherwise the standard identity map.
        AffineMap unitPermMap = (srcRank == 0 && unitVecTy.getRank() > 0)
                                  ? getRank0BroadcastPermMap(b, unitVecTy.getRank())
                                  : b.getMultiDimIdentityMap(unitVecTy.getRank());
        auto unitInBounds = b.getBoolArrayAttr(SmallVector<bool>(unitVecTy.getRank(), true));
        Value res = b.create<vector::TransferReadOp>(loc, unitVecTy, remap(rd.getSource()), unitIndices, unitPermMap,
                                                     remap(rd.getPadding()), Value(), unitInBounds);
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
      // Rank-0 destination: extract the leading lane and write it back with a
      // rank-0 `vector.transfer_write`
      if (srcRank == 0) {
        emitRank0TransferWrite(b, loc, remap(wr.getVector()), remap(wr.getSource()));
        return success();
      }
      b.create<vector::TransferWriteOp>(loc, Type(), remap(wr.getVector()), remap(wr.getSource()),
                                        trailingIndices(srcRank), permMapForRank(srcRank), mask, inBounds);
      return success();
    }
    if (auto bc = dyn_cast<npuv::BroadcastOp>(op)) {
      Type elemType = llvm::cast<npuv::NPUVectorType>(bc.getResult().getType()).getElementType();
      Value srcVal = remap(bc.getSource());
      if (llvm::isa<VectorType>(srcVal.getType()) &&
          llvm::cast<VectorType>(srcVal.getType()).getShape() == ArrayRef<int64_t>({laneCount})) {
        map.map(bc.getResult(), srcVal);
        return success();
      }
      Value res = b.create<vector::BroadcastOp>(loc, vecTypeOf(elemType), srcVal);
      map.map(bc.getResult(), res);
      return success();
    }
    // npuvector unary casts feeding the reduction (e.g. `npuvector.extf` widening
    // a bf16 read up to the f32 accumulator before squaring + summing)
    if (isUnaryNpuvCast(op)) {
      Type outElem = llvm::cast<npuv::NPUVectorType>(op->getResult(0).getType()).getElementType();
      auto outTy = vecTypeOf(outElem);
      Value in = remap(op->getOperand(0));
      Value res = emitVectorCastFromNpuv(b, loc, op, outTy, in);
      if (!res) {
        return op->emitError("npuvector-to-vector: unsupported npuvector cast in reduction producer");
      }
      map.map(op->getResult(0), res);
      return success();
    }
    // Fall back to the generic dim-agnostic emission (cmp / select / cast /
    // arith / math / constant) at register width.
    return tileReductionProducerFallback(b, op, map, laneCount);
  }

  // `tileReductionProducer` helper: emit the generic dim-agnostic producer ops
  // (cmp / select / cast / arith / math / constant) at register width.
  LogicalResult tileReductionProducerFallback(OpBuilder &b, Operation *op, IRMapping &map, int64_t laneCount) const {
    Location loc = op->getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    TileCtx ctx;
    ctx.loopRank = 1;
    ctx.laneCount = laneCount;
    static const llvm::DenseSet<Value> kEmptyUnit;
    static const DimMap kEmptyMap;
    ctx.dimMap = &kEmptyMap;
    ctx.unitVals = &kEmptyUnit;
    // For these ops widthOf falls back to laneCount only when reg-tiled; with an
    // empty map it returns 1, which is wrong here, so emit directly at lanes.
    if (auto ci = dyn_cast<npuv::CmpIOp>(op)) {
      map.map(ci.getResult(), b.create<arith::CmpIOp>(loc, ci.getPredicate(), remap(ci.getLhs()), remap(ci.getRhs())));
      return success();
    }
    if (auto cf = dyn_cast<npuv::CmpFOp>(op)) {
      map.map(cf.getResult(), b.create<arith::CmpFOp>(loc, cf.getPredicate(), remap(cf.getLhs()), remap(cf.getRhs())));
      return success();
    }
    if (auto sel = dyn_cast<npuv::SelectOp>(op)) {
      map.map(sel.getResult(), b.create<arith::SelectOp>(loc, remap(sel.getCondition()), remap(sel.getTrueValue()),
                                                         remap(sel.getFalseValue())));
      return success();
    }
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(cst.getType());
      if (!nvt) {
        map.map(cst.getResult(), b.clone(*op)->getResult(0));
        return success();
      }
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat()) {
        return op->emitError("npuvector-to-vector: only splat npuvector constants are supported");
      }
      auto vecType = VectorType::get({laneCount}, nvt.getElementType());
      map.map(cst.getResult(),
              b.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vecType, dense.getSplatValue<Attribute>())));
      return success();
    }
    // Generic arith/math at lanes width.
    SmallVector<Type> resultTypes;
    for (Value r : op->getResults()) {
      if (auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(r.getType())) {
        resultTypes.push_back(VectorType::get({laneCount}, nvt.getElementType()));
      } else {
        resultTypes.push_back(r.getType());
      }
    }
    SmallVector<Value> operands(op->getNumOperands());
    std::transform(op->operand_begin(), op->operand_end(), operands.begin(), [&](Value v) { return remap(v); });
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

  LogicalResult tileFullReductionKernel(func::FuncOp vfFunc, const KernelPlan &plan) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();
    int64_t laneCount = plan.laneCount;

    npuv::ReductionOp redOp;
    npuv::TransferReadOp refRead;
    SmallVector<Operation *> producers;
    if (failed(analyzeReductionKernel(vfFunc, redOp, refRead, producers))) {
      return failure();
    }
    auto nvt = llvm::cast<npuv::NPUVectorType>(refRead.getVector().getType());
    int64_t rank = nvt.getRank();

    // Trace the post-reduction delivery chain
    ReductionDelivery delivery;
    if (!traceReductionDelivery(redOp, delivery)) {
      return redOp.emitError("npuvector-to-vector: unable to resolve full-reduction delivery chain");
    }
    DeliveryTarget target = resolveDeliveryTarget(delivery);
    if (!target.memref) {
      return redOp.emitError("npuvector-to-vector: full-reduction delivery target is not a memref");
    }

    auto kind = redOp.getKind();
    // The reduction's element type drives the accumulator width; it can differ
    // from the *output* memref element type
    Type scalarElem = getElementTypeOrSelf(redOp.getDest().getType());
    auto regVecTy = VectorType::get({laneCount}, scalarElem);

    OpBuilder builder(redOp);
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cLane = builder.create<arith::ConstantIndexOp>(loc, laneCount);

    SmallVector<Value> bounds;
    bounds.reserve(rank);
    if (failed(collectTileBounds({builder, loc}, nvt, refRead.getDynamicSizes(), refRead.getSource(), bounds))) {
      return failure();
    }
    if (static_cast<int64_t>(bounds.size()) != rank) {
      return failure();
    }

    Value regIdent = buildIdentityAccumulator(builder, loc, kind, regVecTy);
    TypedAttr idAttr = getCombiningIdentityAttr(builder, kind, scalarElem);

    // Accumulator (seed) shape.  we materialize the seed as a rank-0
    // `vector<scalarElem>` so the post-loop sequence is the canonical
    // `extractelement -> multi_reduction -> broadcast` chain.
    auto rank0Ty = VectorType::get({}, scalarElem);
    FullReductionInfo info{redOp, kind, scalarElem, producers, regIdent, regVecTy, laneCount};
    SeedResult seed;
    BuilderEnv benv{builder, loc};
    DeliveryCtx dctx{idAttr, target, delivery, c0};
    buildFullReductionSeed(benv, info, dctx, seed);
    bool reuseTargetAsAccBuf = seed.reuseTargetAsAccBuf;
    Value accBuf = seed.accBuf;
    Value idScalar = seed.idScalar;
    Value seedVec = seed.seedVec;

    Value vecAcc;
    if (failed(buildFullReductionLoopNest({builder, loc}, {rank, bounds}, {c0, c1, cLane},
                                          info, vecAcc))) {
      return failure();
    }

    OpBuilder post(redOp);
    // Pull the seed scalar out of its rank-0 carrier. Legacy keeps the
    // memref<1xT> + transfer_read shape; the new path goes straight from the
    // rank-0 vector constant (or broadcast acc) into `extractelement`.
    Value seedScalarVec;
    if (reuseTargetAsAccBuf) {
      auto rank0Map = AffineMap::get(1, 0, post.getContext());
      seedScalarVec = post.create<vector::TransferReadOp>(loc, rank0Ty, accBuf, ValueRange{c0}, rank0Map, idScalar,
                                                          Value(), post.getBoolArrayAttr({}));
    } else {
      seedScalarVec = seedVec;
    }
    Value seedV = post.create<vector::ExtractElementOp>(loc, seedScalarVec);
    SmallVector<bool> reductionMask(1, true);
    Value reduced = post.create<vector::MultiDimReductionOp>(loc, vecAcc, seedV, reductionMask, kind);

    if (!llvm::isa<npuv::NPUVectorType>(redOp.getResult().getType())) {
      redOp.getResult().replaceAllUsesWith(reduced);
    }

    if (reuseTargetAsAccBuf) {
      if (failed(emitFullReductionLegacyWrite({post, loc}, {reduced, scalarElem}, delivery, accBuf,
                                              {c0, c1, cLane}))) {
        return failure();
      }
    } else {
      if (failed(emitFullReductionNewWrite({post, loc}, {reduced, scalarElem}, target, delivery, c0))) {
        return failure();
      }
    }

    eraseFullReductionKernel(entry, redOp, delivery, producers);
    return success();
  }

  // `tileFullReductionKernel` helper: materialize the accumulator seed. Decides
  // whether the output memref can be reused as the scalar accumulator buffer
  // (legacy `memref<1xT>` path) and, depending on that, sets up either the
  // legacy scalar identity / scratch store or the new rank-0 vector seed.
  void buildFullReductionSeed(const BuilderEnv &env, const FullReductionInfo &info,
                              const DeliveryCtx &dctx, SeedResult &out) const {
    OpBuilder &builder = env.builder;
    Location loc = env.loc;
    npuv::ReductionOp redOp = info.redOp;
    vector::CombiningKind kind = info.kind;
    Type scalarElem = info.scalarElem;
    const DeliveryTarget &target = dctx.target;
    const ReductionDelivery &delivery = dctx.delivery;
    out.reuseTargetAsAccBuf = false;
    if (target.isStore && target.rank == 1 && target.memref) {
      auto mt = llvm::cast<MemRefType>(target.memref.getType());
      llvm::ArrayRef<int64_t> memShape = mt.getShape();
      if (memShape.size() >= 1 && memShape[0] == 1 && mt.getElementType() == scalarElem &&
          delivery.castChain.empty()) {
        out.reuseTargetAsAccBuf = true;
      }
    }
    auto rank0Ty = VectorType::get({}, scalarElem);
    if (out.reuseTargetAsAccBuf) {
      out.idScalar = builder.create<arith::ConstantOp>(loc, dctx.idAttr);
      out.accBuf = target.memref;
      Value seedInit = redOp.getAcc() ? redOp.getAcc() : out.idScalar;
      builder.create<memref::StoreOp>(loc, seedInit, out.accBuf, ValueRange{dctx.c0});
    } else if (Value acc = redOp.getAcc()) {
      out.seedVec = builder.create<vector::BroadcastOp>(loc, rank0Ty, acc);
    } else {
      out.seedVec = buildIdentityAccumulator(builder, loc, kind, rank0Ty);
    }
  }

  // `buildFullReductionLoopNest` helper: emit one register-tiled reduction loop
  // over [lbv, ubv) step lanes, accumulating `lp.initAccs.front()`. When
  // `lp.masked`, the tail lanes are masked (transfer reads padded + a select
  // against the identity) so out-of-range lanes never reach the reduction.
  // Returns the loop result.
  LogicalResult buildReductionRegLoop(BuilderEnv env, PassLoopParams lp, FullReductionInfo info,
                                      Value &result) const {
    if (lp.initAccs.empty()) {
      return info.redOp.emitError("npuvector-to-vector: reduction register loop needs an accumulator");
    }
    OpBuilder &b = env.builder;
    Location loc = env.loc;
    Value init = lp.initAccs.front();
    bool masked = lp.masked;
    auto forOp = b.create<scf::ForOp>(loc, lp.lbv, lp.ubv, lp.cLane, ValueRange{init});
    Value iv = forOp.getInductionVar();
    Value acc = forOp.getRegionIterArgs().front();
    OpBuilder lb = OpBuilder::atBlockBegin(forOp.getBody());

    SmallVector<Value> ivs(lp.outerIvs.begin(), lp.outerIvs.end());
    ivs.push_back(iv);

    Value mask = masked ? tailRegMask(lb, loc, lp.bound, iv, info.laneCount) : Value();
    IRMapping map;
    if (std::any_of(info.producers.begin(), info.producers.end(), [&](Operation *op) {
          return failed(tileReductionProducer(lb, op, map, ivs, mask, info.laneCount));
        })) {
      return failure();
    }
    Value redInput = map.lookupOrDefault(info.redOp.getVector());
    if (redInput.getType() != info.regVecTy) {
      return info.redOp.emitError("npuvector-to-vector: reduction input register type mismatch");
    }
    Value contrib = masked ? lb.create<arith::SelectOp>(loc, mask, redInput, info.regIdent) : redInput;
    Value combined = vector::makeArithReduction(lb, loc, info.kind, acc, contrib);
    lb.create<scf::YieldOp>(loc, combined);
    result = forOp.getResult(0);
    return success();
  }

  // `tileFullReductionKernel` helper: build the register-tiled scf.for nest, tile
  // the reduction producers inside it, accumulate the register reduction and wire
  // up the per-level yields. The innermost (register) dim is split into an
  // aligned main loop and a masked tail loop. Returns the accumulator in
  // `vecAcc`.
  LogicalResult buildFullReductionLoopNest(BuilderEnv env, LoopNestDims dims, LoopConstants lc,
                                           FullReductionInfo info, Value &vecAcc) const {
    const OpBuilder &builder = env.builder;
    Location loc = env.loc;
    int64_t rank = dims.rank;
    ArrayRef<Value> bounds = dims.bounds;
    if (rank < 1 || static_cast<size_t>(rank) > bounds.size()) {
      return failure();
    }
    SmallVector<Value> outerIvs;
    SmallVector<scf::ForOp> outerLoops;
    OpBuilder lb = builder;
    Value curInit = info.regIdent;
    for (int64_t d = 0; d < rank - 1; ++d) {
      auto forOp = lb.create<scf::ForOp>(loc, lc.c0, bounds[d], lc.c1, ValueRange{curInit});
      outerLoops.push_back(forOp);
      outerIvs.push_back(forOp.getInductionVar());
      curInit = forOp.getRegionIterArgs().front();
      lb = OpBuilder::atBlockBegin(forOp.getBody());
    }

    // Split the innermost (register) dim into aligned main + masked tail loops.
    Value bound = bounds[rank - 1];
    RegSplit split = planRegSplit(lb, loc, bound, info.laneCount);
    Value acc = curInit;
    if (split.emitMain) {
      if (failed(buildReductionRegLoop({lb, loc}, {lc.c0, split.alignedBound, bound, lc.cLane,
                                     outerIvs, {acc}, false}, info, acc))) {
        return failure();
      }
    }
    if (split.emitTail) {
      Value tailLb = split.emitMain ? split.alignedBound : lc.c0;
      if (failed(buildReductionRegLoop({lb, loc}, {tailLb, bound, bound, lc.cLane,
                                     outerIvs, {acc}, true}, info, acc))) {
        return failure();
      }
    }

    // Thread the innermost accumulator back up through the outer loops.
    Value innerResult = acc;
    for (int64_t d = rank - 2; d >= 0; --d) {
      OpBuilder yb = OpBuilder::atBlockEnd(outerLoops[d].getBody());
      yb.create<scf::YieldOp>(loc, innerResult);
      innerResult = outerLoops[d].getResult(0);
    }

    vecAcc = innerResult;
    return success();
  }

  // `tileFullReductionKernel` helper: erase the original delivery chain
  // (write -> casts), the reduction, its producers, and any now-dead constants
  // so the unused npuvector values are gone after tiling.
  void eraseFullReductionKernel(Block &entry, npuv::ReductionOp redOp, const ReductionDelivery &delivery,
                                ArrayRef<Operation *> producers) const {
    delivery.finalWrite->erase();
    for (auto it = delivery.castChain.rbegin(); it != delivery.castChain.rend(); ++it) {
      (*it)->erase();
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
  }

  // `tileFullReductionKernel` helper: legacy delivery write -- scalar cast chain,
  // then vector<1xT> + masked transfer_write to memref<1xT> (matches the prior
  // lowering the existing UT depends on).
  LogicalResult emitFullReductionLegacyWrite(BuilderEnv env, ReducedValue rv, const ReductionDelivery &delivery,
                                             Value accBuf, LoopConstants lc) const {
    OpBuilder &post = env.builder;
    Location loc = env.loc;
    Value scalarVal = rv.reduced;
    for (Operation *castOp : delivery.castChain) {
      Value next = emitScalarCastFromNpuv(post, loc, castOp, scalarVal);
      if (!next) {
        return castOp->emitError("npuvector-to-vector: unsupported cast in full-reduction delivery chain");
      }
      scalarVal = next;
    }
    auto out1Ty = VectorType::get({1}, rv.scalarElem);
    Value bcast = post.create<vector::BroadcastOp>(loc, out1Ty, scalarVal);
    Value oneMask = post.create<vector::CreateMaskOp>(loc, VectorType::get({1}, post.getI1Type()), ValueRange{lc.c1});
    post.create<vector::TransferWriteOp>(loc, Type(), bcast, accBuf, ValueRange{lc.c0},
                                         post.getMultiDimIdentityMap(1), oneMask, post.getBoolArrayAttr({true}));
    return success();
  }

  // `tileFullReductionKernel` helper: new delivery write -- broadcast right
  // after the multi_reduction, then apply the npuvector cast chain as vector
  // ops on the broadcast vector. Produces the canonical
  // `multi_reduction -> broadcast -> (vector casts) -> transfer_write` shape
  // with no scratch memref and no single-lane mask on the final write.
  LogicalResult emitFullReductionNewWrite(BuilderEnv env, ReducedValue rv, const DeliveryTarget &target,
                                          const ReductionDelivery &delivery, Value c0) const {
    OpBuilder &post = env.builder;
    Location loc = env.loc;
    Value reduced = rv.reduced;
    Type scalarElem = rv.scalarElem;
    int64_t writeRank = std::max<int64_t>(target.rank, 1);
    auto wrapElem = [&target](Type elem) {
      return (target.rank == 0) ? VectorType::get({}, elem) : VectorType::get({1}, elem);
    };
    Value vecVal = post.create<vector::BroadcastOp>(loc, wrapElem(scalarElem), reduced);
    for (Operation *castOp : delivery.castChain) {
      Type outElem = getElementTypeOrSelf(castOp->getResult(0).getType());
      Value next = emitVectorCastFromNpuv(post, loc, castOp, wrapElem(outElem), vecVal);
      if (!next) {
        return castOp->emitError("npuvector-to-vector: unsupported cast in full-reduction delivery chain");
      }
      vecVal = next;
    }

    if (target.rank == 0) {
      auto rank0WriteMap = AffineMap::get(0, 0, post.getContext());
      post.create<vector::TransferWriteOp>(loc, Type(), vecVal, target.memref, ValueRange{},
                                           rank0WriteMap, Value(), post.getBoolArrayAttr({}));
      return success();
    }
    // Preserve original indices when possible (a `npuv::TransferWriteOp`
    // may write at a runtime index other than 0).
    SmallVector<Value> indices;
    if (auto wr = dyn_cast<npuv::TransferWriteOp>(delivery.finalWrite)) {
      std::copy(wr.getIndices().begin(), wr.getIndices().end(), std::back_inserter(indices));
    } else if (auto st = dyn_cast<memref::StoreOp>(delivery.finalWrite)) {
      std::copy(st.getIndices().begin(), st.getIndices().end(), std::back_inserter(indices));
    }
    if (static_cast<int64_t>(indices.size()) > target.rank) {
      indices.erase(indices.begin(), indices.begin() + (indices.size() - target.rank));
    }
    if (indices.empty()) {
      indices.push_back(c0);
    }
    post.create<vector::TransferWriteOp>(loc, Type(), vecVal, target.memref, indices,
                                         post.getMultiDimIdentityMap(writeRank), Value(),
                                         post.getBoolArrayAttr(SmallVector<bool>(writeRank, true)));
    return success();
  }

  // =========================================================================
  // Partial (last-axis) reduction kernel tiler.
  // =========================================================================

  // Returns true when the vector function contains only last-axis reductions
  // (no transpose) and `loopRank >= 1`, so the partial-reduction tiler can
  // handle it.
  static bool isPartialLastAxisReductionTileable(func::FuncOp vfFunc, int64_t loopRank) {
    if (vfFunc.getBody().empty()) {
      return false;
    }
    Block &entry = vfFunc.getBody().front();
    bool hasReduction = false;
    bool ok = true;
    entry.walk([&](Operation *op) {
      if (isa<npuv::TransposeOp>(op)) {
        ok = false;
      }
      if (auto red = dyn_cast<npuv::ReductionOp>(op)) {
        hasReduction = true;
        auto dims = red.getReductionDims();
        if (!dims || dims->empty()) {
          ok = false;  // full reduction handled elsewhere
          return;
        }
        // Only last-axis reduction is supported here.
        int64_t srcRank = npuVectorRank(red.getVector().getType());
        if (dims->size() != 1 || (*dims)[0] != srcRank - 1) {
          ok = false;
        }
      }
    });
    return hasReduction && ok && loopRank >= 1;
  }

  LogicalResult tilePartialReductionKernel(func::FuncOp vfFunc, const KernelPlan &plan) {
    Block &entry = vfFunc.getBody().front();
    Location loc = vfFunc.getLoc();
    int64_t lanes = plan.laneCount;
    int64_t R = plan.loopRank;
    if (R < 1) {
      return failure();
    }

    // Structure per outer (row) iteration:
    //   * pass 1 over the register/reduction axis with one accumulator iter_arg
    //     per reduction: emit the reduction producers, accumulate, and write any
    //     pure-elementwise outputs that do not depend on a reduction result.
    //   * finalize each reduction to a per-row scalar, then run the per-row
    //     compute.
    //   * pass 2 over the register axis: recompute the register-wide inputs and
    //     emit the outputs that broadcast the per-row results back across the row.

    Value anchor = findAnchorValue(entry);
    if (!anchor) {
      return failure();
    }
    DimMap dimMap;
    llvm::DenseSet<Value> unitVals;
    computeDimMap(entry, anchor, R, {dimMap, unitVals});

    // Classify ops and run the reduction taint analysis.
    SmallVector<npuv::ReductionOp> reductions;
    llvm::DenseSet<Operation *> dependsOnReduction;
    if (failed(collectPartialReductions(entry, reductions, dependsOnReduction))) {
      return failure();
    }

    // Build outer (row) loops first.
    OpBuilder builder(entry.getTerminator());
    SmallVector<Value> bounds;
    if (failed(getPerDimBounds(builder, loc, anchor, bounds)) || static_cast<int64_t>(bounds.size()) != R) {
      return failure();
    }
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value cLane = builder.create<arith::ConstantIndexOp>(loc, lanes);

    // Emit unit ops once before everything.
    IRMapping shared;
    TileCtx baseCtx;
    baseCtx.loopRank = R;
    baseCtx.laneCount = lanes;
    baseCtx.dimMap = &dimMap;
    baseCtx.unitVals = &unitVals;
    SmallVector<Operation *> unitEmitted;
    if (failed(emitUnitOps(builder, entry, shared, baseCtx, unitEmitted))) {
      return failure();
    }

    SmallVector<Value> outerIvs;
    OpBuilder body = builder;
    for (int64_t d = 0; d < R - 1; ++d) {
      auto forOp = body.create<scf::ForOp>(loc, c0, bounds[d], c1);
      outerIvs.push_back(forOp.getInductionVar());
      body = OpBuilder::atBlockBegin(forOp.getBody());
    }

    // Per-reduction identity / accumulators.
    SmallVector<vector::CombiningKind> redKinds;
    SmallVector<Value> redIdents;
    SmallVector<Type> redElems;
    for (auto red : reductions) {
      Type elem = getElementTypeOrSelf(red.getDest().getType());
      redElems.push_back(elem);
      redKinds.push_back(red.getKind());
      redIdents.push_back(buildIdentityAccumulator(body, loc, red.getKind(), VectorType::get({lanes}, elem)));
    }

    // ---- Pass 1: accumulation + pure-elementwise outputs ----
    scf::ForOp pass1;
    ReductionCtx redCtx{reductions, redIdents, redKinds, unitEmitted, &dependsOnReduction};
    if (failed(runPartialPass1({body, entry, shared, baseCtx}, loc,
                               {bounds, outerIvs, c0, cLane, R}, redCtx, pass1))) {
      return failure();
    }

    // ---- Finalize reductions to per-row scalars, then run the per-row compute
    //      (scalar ops not reg-tiled, depending on reductions) ----
    OpBuilder pr = body;  // continue in the outer-row body, after pass1
    pr.setInsertionPointAfter(pass1);
    if (failed(finalizePartialReductions(pr, loc, entry, shared, reductions, redKinds, redElems, pass1, outerIvs,
                                         dimMap, unitVals, unitEmitted, R))) {
      return failure();
    }

    // ---- Pass 2: recompute reg-tiled inputs and emit reduction-dependent
    //      outputs (broadcasting the per-row scalars back across the row) ----
    // Only needed when there is a reduction-dependent, register-tiled output
    // (an output that spreads a per-row scalar back across the row). A pure
    // "reduce and write per-row scalars" kernel has none, so pass 2 would emit
    // nothing but dead register re-reads -- skip it entirely in that case.
    bool needPass2 = false;
    for (Operation &op : entry.without_terminator()) {
      auto wr = dyn_cast<npuv::TransferWriteOp>(&op);
      if (wr && dependsOnReduction.count(&op) && isRegTiled(wr.getVector(), dimMap, unitVals, R)) {
        needPass2 = true;
        break;
      }
    }
    if (needPass2 && failed(runPartialPass2({pr, entry, shared, baseCtx}, loc,
                                            {bounds, outerIvs, c0, cLane, R}, redCtx))) {
      return failure();
    }

    // Erase original compute ops.
    SmallVector<Operation *> computeOps;
    for (Operation &op : entry.without_terminator()) {
      if (touchesNpuVector(&op)) {
        computeOps.push_back(&op);
      }
    }
    eraseComputeOps(entry, computeOps);
    return success();
  }

  // `tilePartialReductionKernel` helper: collect all reduction ops and run the
  // taint analysis (an op depends on a reduction if it transitively uses a
  // reduction result). Fails when the kernel has no reduction.
  LogicalResult collectPartialReductions(Block &entry, SmallVectorImpl<npuv::ReductionOp> &reductions,
                                         llvm::DenseSet<Operation *> &dependsOnReduction) const {
    for (Operation &op : entry) {
      if (auto r = dyn_cast<npuv::ReductionOp>(&op)) {
        reductions.push_back(r);
      }
    }
    if (reductions.empty()) {
      return failure();
    }
    for (Operation &op : entry) {
      bool dep = false;
      for (Value v : op.getOperands()) {
        Operation *def = v.getDefiningOp();
        if (def && (isa<npuv::ReductionOp>(def) || dependsOnReduction.count(def))) {
          dep = true;
          break;
        }
      }
      if (dep) {
        dependsOnReduction.insert(&op);
      }
    }
    return success();
  }

  // `tilePartialReductionKernel` helper: build the pass-1 register loops (with one
  // accumulator iter_arg per reduction). The register dim is split into an
  // aligned main loop and a masked tail loop; the accumulators thread from main
  // into tail. Returns the final loop (whose results are the accumulators) in
  // `pass1`.
  LogicalResult runPartialPass1(const PartialPassEnv &env, Location loc, const LoopIterCtx &iter,
                                const ReductionCtx &red, scf::ForOp &pass1) const {
    if (iter.rank < 1) {
      return failure();
    }
    const ArrayRef<Value> bounds = iter.bounds;
    const size_t innerIdx = static_cast<size_t>(iter.rank - 1);
    if (innerIdx >= bounds.size()) {
      return failure();
    }
    Value bound = bounds[innerIdx];
    RegSplit split = planRegSplit(env.builder, loc, bound, env.baseCtx.laneCount);
    SmallVector<Value> curInit(red.redIdents.begin(), red.redIdents.end());
    bool any = false;
    if (split.emitMain) {
      if (failed(emitPartialPass1Loop(env, loc, red, {iter.c0, split.alignedBound, bound, iter.cLane,
                                     iter.outerIvs, curInit, false}, pass1))) {
        return failure();
      }
      curInit.assign(pass1.getResults().begin(), pass1.getResults().end());
      any = true;
    }
    if (split.emitTail) {
      Value tailLb = split.emitMain ? split.alignedBound : iter.c0;
      if (failed(emitPartialPass1Loop(env, loc, red, {tailLb, bound, bound, iter.cLane,
                                     iter.outerIvs, curInit, true}, pass1))) {
        return failure();
      }
      any = true;
    }
    return success(any);
  }

  // `runPartialPass1` helper: emit one pass-1 register loop over [lbv, ubv).
  LogicalResult emitPartialPass1Loop(PartialPassEnv env, Location loc, ReductionCtx red,
                                     PassLoopParams lp, scf::ForOp &loopOut) const {
    loopOut = env.builder.create<scf::ForOp>(loc, lp.lbv, lp.ubv, lp.cLane, lp.initAccs);
    Value regIv = loopOut.getInductionVar();
    OpBuilder p1 = OpBuilder::atBlockBegin(loopOut.getBody());

    SmallVector<Value> ivs1(lp.outerIvs.begin(), lp.outerIvs.end());
    ivs1.push_back(regIv);

    IRMapping map1;
    // seed with unit values
    for (Operation *u : red.unitEmitted) {
      for (Value r : u->getResults()) {
        if (Value mapped = env.shared.lookupOrDefault(r); mapped != r) {
          map1.map(r, mapped);
        }
      }
    }
    TileCtx ctx1 = env.baseCtx;
    ctx1.ivs = ivs1;
    if (lp.masked) {
      ctx1.mask = tailRegMask(p1, loc, lp.bound, regIv, env.baseCtx.laneCount);
    }

    return emitPartialPass1({p1, env.entry, map1, ctx1}, loc, red, loopOut);
  }

  // `tilePartialReductionKernel` helper: build the pass-2 register loops (aligned
  // main loop + masked tail loop), seed each op map with the per-row scalars /
  // unit values, then emit the pass-2 body in each.
  LogicalResult runPartialPass2(PartialPassEnv env, Location loc, LoopIterCtx iter, ReductionCtx red) const {
    if (iter.rank < 1 || iter.rank > static_cast<int64_t>(iter.bounds.size())) {
      return failure();
    }
    Value bound = iter.bounds[iter.rank - 1];
    RegSplit split = planRegSplit(env.builder, loc, bound, env.baseCtx.laneCount);
    if (split.emitMain) {
      if (failed(emitPartialPass2Loop(env, loc, red, {iter.c0, split.alignedBound, bound, iter.cLane,
                                     iter.outerIvs, {}, false}))) {
        return failure();
      }
    }
    if (split.emitTail) {
      Value tailLb = split.emitMain ? split.alignedBound : iter.c0;
      if (failed(emitPartialPass2Loop(env, loc, red, {tailLb, bound, bound, iter.cLane,
                                     iter.outerIvs, {}, true}))) {
        return failure();
      }
    }
    return success();
  }

  // `runPartialPass2` helper: emit one pass-2 register loop over [lbv, ubv).
  LogicalResult emitPartialPass2Loop(PartialPassEnv env, Location loc, ReductionCtx red,
                                     PassLoopParams lp) const {
    auto pass2 = env.builder.create<scf::ForOp>(loc, lp.lbv, lp.ubv, lp.cLane);
    Value regIv2 = pass2.getInductionVar();
    OpBuilder p2 = OpBuilder::atBlockBegin(pass2.getBody());

    SmallVector<Value> ivs2(lp.outerIvs.begin(), lp.outerIvs.end());
    ivs2.push_back(regIv2);

    IRMapping map2(env.shared);  // per-row scalars + unit values available
    TileCtx ctx2 = env.baseCtx;
    ctx2.ivs = ivs2;
    if (lp.masked) {
      ctx2.mask = tailRegMask(p2, loc, lp.bound, regIv2, env.baseCtx.laneCount);
    }

    return emitPartialPass2({p2, env.entry, map2, ctx2}, red);
  }

  // `tilePartialReductionKernel` helper: pass 1 body -- emit every reg-tiled op
  // that does NOT depend on a reduction (producers feeding reductions + pure
  // elementwise outputs), accumulate each reduction into its iter_arg, and yield
  // the running accumulators.
  LogicalResult emitPartialPass1(PartialPassEnv env, Location loc, ReductionCtx red, scf::ForOp pass1) const {
    OpBuilder &p1 = env.builder;
    IRMapping &map1 = env.shared;
    const TileCtx &ctx1 = env.baseCtx;
    auto regTiled = [this, &ctx1](Value v) { return isRegTiled(v, *ctx1.dimMap, *ctx1.unitVals, ctx1.loopRank); };
    llvm::DenseMap<npuv::ReductionOp, int> redIndex;
    for (auto [i, r] : llvm::enumerate(red.reductions)) {
      redIndex[r] = static_cast<int>(i);
    }
    SmallVector<Value> redAccumulated(red.reductions.size());

    for (Operation &op : env.entry.without_terminator()) {
      if (llvm::is_contained(red.unitEmitted, &op) || !touchesNpuVector(&op)) {
        continue;
      }
      if (auto r = dyn_cast<npuv::ReductionOp>(&op)) {
        Value src = map1.lookupOrDefault(r.getVector());
        int idx = redIndex[r];
        // In the masked tail loop, zero out the inactive lanes (to the reduction
        // identity) so they do not pollute the accumulator.
        Value contrib = ctx1.mask ? p1.create<arith::SelectOp>(loc, ctx1.mask, src, red.redIdents[idx]) : src;
        Value acc = pass1.getRegionIterArgs()[idx];
        redAccumulated[idx] = vector::makeArithReduction(p1, loc, red.redKinds[idx], acc, contrib);
        continue;
      }
      if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        // Pure-elementwise output (independent of any reduction) is written here.
        if (!red.dependsOnReduction->count(&op) && regTiled(wr.getVector())) {
          if (failed(emitTiledOp(p1, &op, map1, ctx1))) {
            return failure();
          }
        }
        continue;
      }
      // Reg-tiled producer not depending on a reduction.
      if (!red.dependsOnReduction->count(&op) && !op.getResults().empty() && regTiled(op.getResult(0))) {
        if (failed(emitTiledOp(p1, &op, map1, ctx1))) {
          return failure();
        }
      }
    }
    // Carry the running accumulators (fall back to incoming iter arg if a
    // reduction was not reached, which should not happen).
    SmallVector<Value> yields(red.reductions.size());
    for (size_t i = 0; i < red.reductions.size(); ++i) {
      yields[i] = redAccumulated[i] ? redAccumulated[i] : pass1.getRegionIterArgs()[i];
    }
    p1.create<scf::YieldOp>(loc, yields);
    return success();
  }

  // `tilePartialReductionKernel` helper: horizontally reduce each register
  // accumulator to a per-row scalar (recording it in `shared`), then emit the
  // per-row scalar compute for the non-reg-tiled, reduction-dependent ops.
  LogicalResult finalizePartialReductions(OpBuilder &pr, Location loc, Block &entry, IRMapping &shared,
                                          ArrayRef<npuv::ReductionOp> reductions,
                                          ArrayRef<vector::CombiningKind> redKinds, ArrayRef<Type> redElems,
                                          scf::ForOp pass1, ValueRange outerIvs, const DimMap &dimMap,
                                          const llvm::DenseSet<Value> &unitVals, ArrayRef<Operation *> unitEmitted,
                                          int64_t R) const {
    auto regTiled = [this, &dimMap, &unitVals, R](Value v) { return isRegTiled(v, dimMap, unitVals, R); };
    for (size_t i = 0; i < reductions.size(); ++i) {
      auto red = reductions[i];
      Value acc = pass1.getResult(i);
      // Horizontal reduce the register accumulator (vector<lanes>) to a scalar,
      // using the exact same shape as the full-reduction path: the seed must be
      // an `extractelement` of a rank-0 vector (not a bare scalar constant), so
      // the downstream lowering recognizes the scalar-reduction pattern.
      auto rank0Ty = VectorType::get({}, redElems[i]);
      Value rank0Ident = buildIdentityAccumulator(pr, loc, redKinds[i], rank0Ty);
      Value seed = pr.create<vector::ExtractElementOp>(loc, rank0Ident);
      SmallVector<bool> rmask(1, true);
      Value scalar = pr.create<vector::MultiDimReductionOp>(loc, acc, seed, rmask, redKinds[i]);
      // Per-row values are represented as scalars; record into shared map.
      shared.map(red.getResult(), scalar);
    }

    for (Operation &op : entry.without_terminator()) {
      if (llvm::is_contained(unitEmitted, &op) || !touchesNpuVector(&op)) {
        continue;
      }
      if (isa<npuv::ReductionOp>(&op)) {
        continue;
      }
      if (op.getResults().empty()) {
        continue;
      }
      // per-row scalar op: result not reg-tiled (i.e. a row scalar).
      if (!regTiled(op.getResult(0))) {
        if (failed(emitPerRowScalarOp(pr, &op, shared, outerIvs, dimMap))) {
          return failure();
        }
      }
    }

    // Emit the per-row (non-reg-tiled) reduction-result writes: a partial
    // reduction that folds the register axis yields one scalar per outer row,
    // which is stored back with a shape-1 `vector.transfer_write` at the row
    // index (the downstream lowering does not accept `memref.store`).
    for (Operation &op : entry.without_terminator()) {
      if (llvm::is_contained(unitEmitted, &op) || !touchesNpuVector(&op)) {
        continue;
      }
      auto wr = dyn_cast<npuv::TransferWriteOp>(&op);
      if (!wr || regTiled(wr.getVector())) {
        continue;
      }
      if (failed(emitPerRowScalarWrite(pr, wr, shared, outerIvs, dimMap))) {
        return failure();
      }
    }
    return success();
  }

  // `finalizePartialReductions` helper: store one per-row scalar back to its
  // memref with a shape-1 `vector.transfer_write` at the row index.
  LogicalResult emitPerRowScalarWrite(OpBuilder &b, npuv::TransferWriteOp wr, IRMapping &map, ValueRange outerIvs,
                                      const DimMap &dimMap) const {
    Location loc = wr.getLoc();
    Value scalar = map.lookupOrDefault(wr.getVector());
    if (llvm::isa<ShapedType>(scalar.getType())) {
      return wr.emitError("npuvector-to-vector: per-row reduction result is not a scalar");
    }
    Type elem = scalar.getType();
    ArrayRef<int64_t> m =
      dimMap.count(wr.getVector()) ? ArrayRef<int64_t>(dimMap.find(wr.getVector())->second) : ArrayRef<int64_t>();
    SmallVector<Value> indices;
    for (size_t k = 0; k < wr.getIndices().size(); ++k) {
      Value base = map.lookupOrDefault(wr.getIndices()[k]);
      Value iv;
      if (k < m.size() && m[k] >= 0 && m[k] < static_cast<int64_t>(outerIvs.size())) {
        iv = outerIvs[m[k]];
      }
      indices.push_back(addIndex(b, loc, base, iv));
    }
    int64_t srcRank = getShapedSourceRank(wr.getSource());
    if (srcRank >= 0 && static_cast<int64_t>(indices.size()) > srcRank) {
      indices.erase(indices.begin(), indices.begin() + (indices.size() - srcRank));
    }
    if (srcRank == 0) {
      emitRank0TransferWrite(b, loc, b.create<vector::BroadcastOp>(loc, VectorType::get({1}, elem), scalar),
                             map.lookupOrDefault(wr.getSource()));
      return success();
    }
    auto vec1Ty = VectorType::get({1}, elem);
    Value vec1 = b.create<vector::BroadcastOp>(loc, vec1Ty, scalar);
    auto permMap = AffineMap::getMinorIdentityMap(static_cast<unsigned>(indices.size()), 1, b.getContext());
    b.create<vector::TransferWriteOp>(loc, Type(), vec1, map.lookupOrDefault(wr.getSource()), indices, permMap,
                                      Value(), b.getBoolArrayAttr({true}));
    return success();
  }

  // `tilePartialReductionKernel` helper: pass 2 body -- recompute reg-tiled
  // inputs and emit the reduction-dependent outputs (broadcasting the per-row
  // scalars back across the row).
  LogicalResult emitPartialPass2(PartialPassEnv env, ReductionCtx red) const {
    OpBuilder &p2 = env.builder;
    IRMapping &map2 = env.shared;
    const TileCtx &ctx2 = env.baseCtx;
    auto regTiled = [this, &ctx2](Value v) { return isRegTiled(v, *ctx2.dimMap, *ctx2.unitVals, ctx2.loopRank); };
    for (Operation &op : env.entry.without_terminator()) {
      if (llvm::is_contained(red.unitEmitted, &op) || !touchesNpuVector(&op)) {
        continue;
      }
      if (isa<npuv::ReductionOp>(&op)) {
        continue;
      }
      if (op.getResults().empty()) {
        // a transfer_write
      }
      if (auto wr = dyn_cast<npuv::TransferWriteOp>(&op)) {
        if (red.dependsOnReduction->count(&op) && regTiled(wr.getVector())) {
          if (failed(emitTiledOp(p2, &op, map2, ctx2))) {
            return failure();
          }
        }
        continue;
      }
      // Reg-tiled op: either a reduction-dependent compute (needed for pass2) or
      // a pure input that pass2 must recompute (because its pass1 value lives in
      // a different loop). Emit if reg-tiled and not already mapped.
      if (!op.getResults().empty() && regTiled(op.getResult(0))) {
        if (!map2.contains(op.getResult(0))) {
          if (failed(emitTiledOp(p2, &op, map2, ctx2))) {
            return failure();
          }
        }
      }
    }
    return success();
  }

  // `emitPerRowScalarOp` helper: lower a per-row `npuv::TransferReadOp` into a
  // shape-1 `vector.transfer_read` at the row index, then extract the scalar
  // lane for the downstream scalar ops.
  LogicalResult emitPerRowTransferRead(OpBuilder &b, npuv::TransferReadOp rd, IRMapping &map, ValueRange outerIvs,
                                       const DimMap &dimMap) const {
    Location loc = rd.getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };
    auto vt = llvm::cast<npuv::NPUVectorType>(rd.getVector().getType());
    Type elem = vt.getElementType();
    ArrayRef<int64_t> m =
      dimMap.count(rd.getResult()) ? ArrayRef<int64_t>(dimMap.find(rd.getResult())->second) : ArrayRef<int64_t>();
    SmallVector<Value> indices;
    for (size_t k = 0; k < rd.getIndices().size(); ++k) {
      Value base = remap(rd.getIndices()[k]);
      Value iv;
      if (k < m.size() && m[k] >= 0 && m[k] < static_cast<int64_t>(outerIvs.size())) {
        iv = outerIvs[m[k]];
      }
      indices.push_back(addIndex(b, loc, base, iv));
    }
    // Drop leading placeholder indices that exceed the actual source rank
    // (npuvector+scalar form may emit a `[%c0]` even on a rank-0 memref).
    int64_t srcRank = getShapedSourceRank(rd.getSource());
    if (srcRank >= 0 && static_cast<int64_t>(indices.size()) > srcRank) {
      indices.erase(indices.begin(), indices.begin() + (indices.size() - srcRank));
    }
    // Read the single element with a shape-1 `vector.transfer_read` (never
    // memref.load) and extract the scalar lane for the downstream scalar
    // ops. We avoid emitting a shapeless rank-0 vector here so every
    // vector value in the lowered IR carries an explicit shape; for a
    // rank-0 source memref the read uses a `() -> (0)` broadcast
    // permutation map, otherwise the standard minor-identity map.
    auto vec1Ty = VectorType::get({1}, elem);
    AffineMap permMap;
    if (indices.empty()) {
      permMap = AffineMap::get(0, 0, b.getAffineConstantExpr(0), b.getContext());
    } else {
      permMap = AffineMap::getMinorIdentityMap(static_cast<unsigned>(indices.size()), 1, b.getContext());
    }
    Value vec1 =
      b.create<vector::TransferReadOp>(loc, vec1Ty, remap(rd.getSource()), indices, permMap, remap(rd.getPadding()),
                                       Value(), b.getBoolArrayAttr({true}));
    Value scalar = b.create<vector::ExtractOp>(loc, vec1, ArrayRef<int64_t>{0});
    // Represent per-row value as a scalar; downstream scalar ops use it.
    map.map(rd.getResult(), scalar);
    return success();
  }

  // Emit a per-row (scalar) op: reads become memref.load at the row index,
  // elementwise/cmp/select/cast become scalar arith. Broadcast of a scalar to a
  // per-row scalar stays a scalar (identity).
  LogicalResult emitPerRowScalarOp(OpBuilder &b, Operation *op, IRMapping &map, ValueRange outerIvs,
                                   const DimMap &dimMap) const {
    Location loc = op->getLoc();
    auto remap = [&map](Value v) { return map.lookupOrDefault(v); };

    if (auto rd = dyn_cast<npuv::TransferReadOp>(op)) {
      return emitPerRowTransferRead(b, rd, map, outerIvs, dimMap);
    }
    if (auto bc = dyn_cast<npuv::BroadcastOp>(op)) {
      // scalar -> per-row scalar: identity passthrough.
      map.map(bc.getResult(), remap(bc.getSource()));
      return success();
    }
    if (auto ci = dyn_cast<npuv::CmpIOp>(op)) {
      map.map(ci.getResult(), b.create<arith::CmpIOp>(loc, ci.getPredicate(), remap(ci.getLhs()), remap(ci.getRhs())));
      return success();
    }
    if (auto cf = dyn_cast<npuv::CmpFOp>(op)) {
      map.map(cf.getResult(), b.create<arith::CmpFOp>(loc, cf.getPredicate(), remap(cf.getLhs()), remap(cf.getRhs())));
      return success();
    }
    if (auto sel = dyn_cast<npuv::SelectOp>(op)) {
      map.map(sel.getResult(), b.create<arith::SelectOp>(loc, remap(sel.getCondition()), remap(sel.getTrueValue()),
                                                         remap(sel.getFalseValue())));
      return success();
    }
    // scalar casts: reuse the shared `emitCastFromNpuv` dispatch (covers all
    // unary npuvector casts, not just the six historically listed here).
    if (isUnaryNpuvCast(op)) {
      Type outElem = getElementTypeOrSelf(op->getResult(0).getType());
      Value casted = emitCastFromNpuv(b, loc, op, outElem, remap(op->getOperand(0)));
      if (!casted) {
        return op->emitError("npuvector-to-vector: unsupported per-row cast in partial reduction kernel");
      }
      map.map(op->getResult(0), casted);
      return success();
    }

    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      auto nvt = llvm::dyn_cast<npuv::NPUVectorType>(cst.getType());
      if (!nvt) {
        map.map(cst.getResult(), b.clone(*op)->getResult(0));
        return success();
      }
      auto dense = llvm::dyn_cast<DenseElementsAttr>(cst.getValue());
      if (!dense || !dense.isSplat()) {
        return op->emitError("npuvector-to-vector: only splat npuvector constants are supported");
      }
      map.map(cst.getResult(), b.create<arith::ConstantOp>(loc, cast<TypedAttr>(dense.getSplatValue<Attribute>())));
      return success();
    }

    if (llvm::isa_and_nonnull<arith::ArithDialect, math::MathDialect>(op->getDialect())) {
      SmallVector<Type> resultTypes;
      std::transform(op->result_begin(), op->result_end(), std::back_inserter(resultTypes),
                     [](Value r) { return getElementTypeOrSelf(r.getType()); });
      SmallVector<Value> operands(op->getNumOperands());
      std::transform(op->operand_begin(), op->operand_end(), operands.begin(), [&](Value v) { return remap(v); });
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
    return op->emitError("npuvector-to-vector: unsupported per-row op in partial reduction kernel");
  }

  // -------------------------------------------------------------------------
  // Fallback: direct per-op type conversion (no register tiling). Used only for
  // kernels that match no tileable category.
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

  // =========================================================================
  // Step 1: convertNpuVectorToVector -- compute lane count, kernel category and
  // per-op loop layers. This is the planning step; it determines how each op
  // converts (reduction -> multi_reduction, broadcast honors `dimension`) and
  // how many register loop layers it needs.
  // =========================================================================
  LogicalResult convertNpuVectorToVector(func::FuncOp vfFunc, KernelPlan &plan) {
    func::FuncOp archFunc = findArchFunc(vfFunc);
    std::optional<int64_t> regWidth = getRegisterWidthBytes(archFunc);
    if (!regWidth) {
      vfFunc.emitError(
        "cannot determine register vector width: missing 'arch' attribute on the kernel or any of its callers");
      return failure();
    }
    plan.laneCount = computeLaneCount(vfFunc, *regWidth);

    if (vfFunc.getBody().empty()) {
      plan.category = KernelCategory::Fallback;
      return success();
    }
    Block &entry = vfFunc.getBody().front();

    // Loop layers = the iteration-space rank (highest-rank npuvector transfer).
    Value anchor = findAnchorValue(entry);
    plan.loopRank = anchor ? npuVectorRank(anchor.getType()) : 0;

    // Per-op loop layers metadata.
    for (Operation &op : entry.without_terminator()) {
      if (!touchesNpuVector(&op)) {
        continue;
      }
      int64_t layers = 0;
      for (Value r : op.getResults()) {
        layers = std::max<int64_t>(layers, npuVectorRank(r.getType()));
      }
      plan.opLayers[&op] = layers;
    }

    // Category detection.
    bool hasTranspose = false;
    entry.walk([&](npuv::TransposeOp) { hasTranspose = true; });
    if (hasTranspose) {
      plan.category = KernelCategory::Transpose;
      return success();
    }
    if (isFullReductionTileable(vfFunc)) {
      plan.category = KernelCategory::FullReduction;
      return success();
    }
    bool hasReduction = false;
    entry.walk([&](npuv::ReductionOp) { hasReduction = true; });
    if (hasReduction) {
      plan.category = isPartialLastAxisReductionTileable(vfFunc, plan.loopRank) ? KernelCategory::PartialReduction
                                                                                : KernelCategory::Fallback;
      return success();
    }
    plan.category = (anchor && plan.loopRank > 0) ? KernelCategory::Elementwise : KernelCategory::Fallback;
    return success();
  }

  // =========================================================================
  // Step 2: tileVectorToRegister -- emit the register-fitted scf.for nest using
  // the plan from step 1.
  // =========================================================================
  LogicalResult tileVectorToRegister(func::FuncOp vfFunc, const KernelPlan &plan) {
    switch (plan.category) {
      case KernelCategory::Elementwise:
        return tileElementwiseKernel(vfFunc, plan);
      case KernelCategory::FullReduction:
        return tileFullReductionKernel(vfFunc, plan);
      case KernelCategory::PartialReduction:
        return tilePartialReductionKernel(vfFunc, plan);
      case KernelCategory::Transpose:
        return tileTransposeKernel(vfFunc, plan);
      case KernelCategory::Fallback:
      default:
        return directConvert(vfFunc, plan.laneCount);
    }
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<func::FuncOp> vfFuncs;
    module.walk([&](func::FuncOp f) {
      if (f->hasAttr(kVectorFunctionAttr)) {
        vfFuncs.push_back(f);
      }
    });

    for (func::FuncOp f : vfFuncs) {
      // Step 1: plan the conversion (lane count, loop layers, category).
      KernelPlan plan;
      if (failed(convertNpuVectorToVector(f, plan))) {
        signalPassFailure();
        return;
      }
      // Step 2: generate the register-sized scf.for nest.
      if (failed(tileVectorToRegister(f, plan))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createNPUVectorToVectorPass() {
  return std::make_unique<NPUVectorToVector>();
}
}  // namespace mlir
