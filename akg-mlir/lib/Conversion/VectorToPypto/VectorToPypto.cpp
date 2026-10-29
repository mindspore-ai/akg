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

#include "akg/Conversion/VectorToPypto/VectorToPypto.h"

#include <iomanip>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

#include "akg/Dialect/EmitPy/IR/EmitPy.h"
#include "akg/Dialect/EmitPy/IR/EmitPyOps.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "akg/Dialect/Pypto/IR/PyptoTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;  // NOLINT(build/namespaces)

namespace emitpy = mlir::emitpy;
namespace npuvector = mlir::npuvector;
namespace pypto = mlir::pypto;

namespace mlir {

#define GEN_PASS_DEF_CONVERTVECTORTOPYPTO
#include "akg/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PyPTO type mapping
//===----------------------------------------------------------------------===//

/// Maps an MLIR scalar element type onto the PyPTO dtype name, e.g.
/// f32 -> "pl.FP32", i32 -> "pl.INT32".
static std::optional<std::string> pyptoDtypeName(Type type) {
  if (isa<FloatType>(type)) {
    if (type.isF16()) {
      return std::string("pl.FP16");
    }
    if (type.isBF16()) {
      return std::string("pl.BF16");
    }
    if (type.isF32()) {
      return std::string("pl.FP32");
    }
    return std::nullopt;  // f64 and friends have no PyPTO dtype
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    bool unsignedInt = intType.isUnsigned();
    switch (intType.getWidth()) {
      case 1:
        return std::string("pl.BOOL");
      case 8:
        return std::string(unsignedInt ? "pl.UINT8" : "pl.INT8");
      case 16:
        return std::string(unsignedInt ? "pl.UINT16" : "pl.INT16");
      case 32:
        return std::string(unsignedInt ? "pl.UINT32" : "pl.INT32");
      case 64:
        return std::string(unsignedInt ? "pl.UINT64" : "pl.INT64");
      default:
        return std::nullopt;
    }
  }
  if (isa<IndexType>(type)) {
    return std::string("pl.INT64");
  }
  return std::nullopt;
}

/// The type converter is deliberately the identity: the conversion
/// framework would otherwise materialize unrealized_conversion_cast ops for
/// every operand whose type maps to something else, and a full conversion
/// rejects those. Values inside the kernel keep their source types; only the
/// function signature is rewritten by hand, into opaque PyPTO annotation
/// types, by the downstream convert-pypto-to-emitpy pass.
class VectorToPyptoTypeConverter : public TypeConverter {
 public:
  explicit VectorToPyptoTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
  }
};

//===----------------------------------------------------------------------===//
// Python text helpers
//===----------------------------------------------------------------------===//

/// Renders a floating-point value as Python source text.
static std::string floatToPythonText(APFloat value) {
  if (value.isInfinity() || value.isNaN()) {
    // Non-finite values have no Python numeric literal, and the PIL parser
    // rejects a float('inf') call ("Unsupported function call") wherever it
    // appears (pl.const argument, tile.full fill value). Spell the exact
    // IEEE 754 binary64 bit pattern with the lossless dump form instead.
    bool losesInfo = false;
    value.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &losesInfo);
    // Inf/NaN exponent bits are all ones, so the unsigned hexadecimal form is
    // always exactly the 16 digits the dump spelling requires.
    SmallString<32> bits;
    value.bitcastToAPInt().toString(bits, 16, false, false, false);
    return "pl._dump.float64('" + std::string(bits.begin(), bits.end()) + "')";
  }
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value.convertToDouble();
  std::string text = oss.str();
  if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
      text.find('E') == std::string::npos) {
    text += ".0";
  }
  return text;
}

/// Renders an integer value as Python source text.
static std::optional<std::string> intToPythonText(const APInt &value) {
  if (value.getSignificantBits() > 64) {
    return std::nullopt;
  }
  return std::to_string(value.getSExtValue());
}

/// Extracts the integer value of a constant index operand: index constants
/// stay `arith.constant` through this stage (only convert-pypto-to-emitpy
/// lowers them to `emitpy.literal`).
static std::optional<int64_t> constIndexValue(Value idx) {
  if (auto constOp = idx.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

/// PTOAS allocates tiles in whole 32-byte rows: a constant tile extent is
/// rounded up so `extent * elemBytes` is a multiple of 32. Only the
/// allocation is padded -- the valid shape still bounds the data actually
/// moved and computed, so tail iterations stay within the source buffer.
static int64_t alignExtentTo32B(int64_t extent, unsigned elemBytes) {
  if (elemBytes == 0 || extent <= 0) {
    return extent;
  }
  int64_t bytes = extent * static_cast<int64_t>(elemBytes);
  return llvm::alignTo(bytes, 32) / static_cast<int64_t>(elemBytes);
}

/// Extracts the constant integer of a mixed static/dynamic entry, or nullopt
/// when the entry is a runtime value.
static std::optional<int64_t> staticIntValue(OpFoldResult value) {
  if (!value.is<Attribute>()) {
    return std::nullopt;
  }
  auto intAttr = dyn_cast<IntegerAttr>(value.get<Attribute>());
  if (!intAttr) {
    return std::nullopt;
  }
  return intAttr.getInt();
}

/// Extracts the integer value of a mixed static/dynamic bound or step, as
/// exposed by scf.forall: an IntegerAttr is the static form, a value (a
/// constant op) the dynamic one.
static std::optional<int64_t> mixedIndexValue(OpFoldResult bound) {
  if (bound.is<Attribute>()) {
    auto intAttr = dyn_cast<IntegerAttr>(bound.get<Attribute>());
    if (!intAttr) {
      return std::nullopt;
    }
    return intAttr.getInt();
  }
  return constIndexValue(bound.get<Value>());
}

/// Replaces a view op with `pypto.slice`: the window is full-rank over the
/// source, with unit dims standing in for the axes a rank-reducing view
/// drops (an empty `droppedDims` keeps the op rank-preserving). Mixed
/// static/dynamic offsets are materialized as index constants, which the
/// main conversion turns into `emitpy.literal`; convert-pypto-to-emitpy
/// folds them back into the `pl.slice` offsets list.
static void replaceWithSliceOp(Operation *op, Value source, Value result, ArrayRef<int64_t> window,
                               ArrayRef<OpFoldResult> offsets, ArrayRef<unsigned> droppedDims,
                               ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  SmallVector<Value> offsetValues;
  offsetValues.reserve(offsets.size());
  for (OpFoldResult offset : offsets) {
    if (auto cst = staticIntValue(offset)) {
      offsetValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, *cst));
    } else {
      offsetValues.push_back(offset.get<Value>());
    }
  }
  DenseI64ArrayAttr dropDimsAttr;
  if (!droppedDims.empty()) {
    dropDimsAttr = rewriter.getDenseI64ArrayAttr(llvm::to_vector_of<int64_t>(droppedDims));
  }
  // The replacement keeps the original result type: the type converter is
  // the identity, so a type-changing replacement would demand a
  // materialization at already-converted users and abort the full
  // conversion.
  rewriter.replaceOpWithNewOp<pypto::SliceOp>(op, result.getType(), source, window, ValueRange(offsetValues),
                                              dropDimsAttr);
}

/// Creates a scratch tile (`pypto.create_tile`); convert-pypto-to-emitpy
/// lowers it into `pl.create_tile(shape, dtype=...,
/// target_memory=pl.MemorySpace.Vec)`.
static Value createTmpTile(ConversionPatternRewriter &rewriter, Location loc, ArrayRef<int64_t> shape, Type elemType) {
  assert(pyptoDtypeName(elemType) && "caller validated the element type");
  return rewriter.create<pypto::CreateTileOp>(loc, pypto::TileType::get(rewriter.getContext(), elemType, shape))
    .getResult();
}

static std::optional<SmallVector<int64_t>> tileAllocationShape(Value v, unsigned depth = 0);

/// Statically-known bounds (`maxSizes`/`paddedShapes`) padded to whole
/// 32-byte rows the way the allocation will be; fails when a bound is
/// dynamic or the arity does not match the type's rank.
static std::optional<SmallVector<int64_t>> alignAllocationBounds(Type type, ValueRange bounds) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || bounds.size() != static_cast<size_t>(shaped.getRank())) {
    return std::nullopt;
  }
  unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<int64_t> aligned;
  aligned.reserve(bounds.size());
  for (Value bound : bounds) {
    auto value = constIndexValue(bound);
    if (!value) {
      return std::nullopt;
    }
    aligned.push_back(alignExtentTo32B(*value, elemBytes));
  }
  return aligned;
}

/// A partial reduction's result keeps the input's allocation on the
/// surviving axes: the folded axis is 1 in the keepdim form, or dropped
/// by the incremental lowering's rank-reduced form.
static std::optional<SmallVector<int64_t>> allocationShapeOfReduction(Value result, pypto::ReductionOp reduction,
                                                                     unsigned depth) {
  auto bounds = tileAllocationShape(reduction.getVector(), depth + 1);
  auto shaped = dyn_cast<ShapedType>(result.getType());
  if (!bounds || !shaped) {
    return std::nullopt;
  }
  size_t folded = reduction.getAxis() == pypto::ReduceAxis::row ? bounds->size() - 1 : 0;
  if (shaped.getRank() == static_cast<int64_t>(bounds->size())) {
    SmallVector<int64_t> kept(*bounds);
    kept[folded] = 1;
    return kept;
  }
  if (shaped.getRank() + 1 == static_cast<int64_t>(bounds->size())) {
    SmallVector<int64_t> kept(*bounds);
    kept.erase(kept.begin() + folded);
    return kept;
  }
  return std::nullopt;
}

/// A loop result keeps the allocation of the value the loop yields for
/// that result.
static std::optional<SmallVector<int64_t>> allocationShapeOfLoopResult(Value v, scf::ForOp loop, unsigned depth) {
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (result == v && yield) {
      return tileAllocationShape(yield.getOperands()[index], depth + 1);
    }
  }
  return std::nullopt;
}

/// The element-wise tile ops, casts, selects and comparisons whose
/// allocation stays the operand's. Both the unconverted and the pypto
/// forms appear because patterns run in arbitrary order.
static bool keepsOperandAllocation(Operation *def) {
  if (isa<pypto::AddOp, pypto::SubOp, pypto::MulOp, pypto::DivOp, pypto::MaximumOp, pypto::MinimumOp, pypto::CastOp,
          pypto::AbsOp, pypto::ExpOp, pypto::LogOp, pypto::SqrtOp, pypto::RsqrtOp, pypto::SinOp, pypto::CosOp,
          arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp, arith::SubIOp, arith::SubFOp, arith::DivSIOp,
          arith::DivUIOp, arith::DivFOp, arith::MaxSIOp, arith::MinSIOp, arith::ExtFOp, arith::TruncFOp,
          arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
          arith::FPToUIOp, npuvector::ExtFOp, npuvector::TruncFOp, npuvector::ExtSIOp, npuvector::ExtUIOp,
          npuvector::TruncIOp, npuvector::SIToFPOp, npuvector::UIToFPOp, npuvector::FPToSIOp, npuvector::FPToUIOp,
          math::AbsFOp, math::ExpOp, math::LogOp, math::SqrtOp, math::RsqrtOp, math::SinOp, math::CosOp,
          math::RoundOp, math::FloorOp, math::CeilOp>(def)) {
    return true;
  }
  // Selects and comparisons keep the operands' allocation too: a dynamic
  // tile fed through a where/select or a comparison before a dynamic
  // reduction or store still names the read's bound.
  return isa<pypto::SelectOp, pypto::CmpOp, npuvector::SelectOp, arith::SelectOp, npuvector::CmpIOp, npuvector::CmpFOp,
             arith::MaximumFOp, arith::MinimumFOp>(def);
}

/// The first shaped operand with a resolvable allocation shape.
static std::optional<SmallVector<int64_t>> allocationShapeFromOperands(Operation *def, unsigned depth) {
  for (Value operand : def->getOperands()) {
    if (!isa<ShapedType>(operand.getType())) {
      continue;
    }
    if (auto bounds = tileAllocationShape(operand, depth + 1)) {
      return bounds;
    }
  }
  return std::nullopt;
}

/// The statically allocatable shape covering a tile: static dims stay and
/// dynamic (`?`) dims resolve to the producing read/broadcast's `maxSizes`
/// bound, padded to whole 32-byte rows the way the allocation will be.
/// Element-wise users, casts and loop-carried accumulators keep the
/// allocation of the read they combine.
static std::optional<SmallVector<int64_t>> tileAllocationShape(Value v, unsigned depth) {
  if (depth > 16) {
    return std::nullopt;
  }
  // Loop-carried tiles: an scf.for result keeps the allocation of the
  // yielded value, and an iter_arg keeps the init's.
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
      if (arg.getArgNumber() > 0) {
        return tileAllocationShape(loop.getInitArgs()[arg.getArgNumber() - 1], depth + 1);
      }
    }
    return std::nullopt;
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }
  if (auto read = dyn_cast<npuvector::TransferReadOp>(def)) {
    return alignAllocationBounds(v.getType(), read.getMaxSizes());
  }
  if (auto load = dyn_cast<pypto::LoadOp>(def)) {
    return alignAllocationBounds(v.getType(), load.getPaddedShapes());
  }
  if (auto broadcast = dyn_cast<npuvector::BroadcastOp>(def)) {
    return alignAllocationBounds(v.getType(), broadcast.getMaxSizes());
  }
  if (auto reduction = dyn_cast<pypto::ReductionOp>(def)) {
    return allocationShapeOfReduction(v, reduction, depth);
  }
  if (auto loop = dyn_cast<scf::ForOp>(def)) {
    return allocationShapeOfLoopResult(v, loop, depth);
  }
  if (keepsOperandAllocation(def)) {
    return allocationShapeFromOperands(def, depth);
  }
  return std::nullopt;
}

/// Views `input` as a `[rows, cols]` tile (`pypto.reshape`);
/// convert-pypto-to-emitpy lowers it into `pl.reshape(input, [rows, cols])`.
static Value reshapeTileOp(ConversionPatternRewriter &rewriter, Location loc, Value input, int64_t rows, int64_t cols,
                           Type elemType) {
  assert(pyptoDtypeName(elemType) && "caller validated the element type");
  return rewriter.create<pypto::ReshapeOp>(
    loc, pypto::TileType::get(rewriter.getContext(), elemType, SmallVector<int64_t>{rows, cols}), input);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Rewrites an operation into the matching pypto op, e.g.
/// `arith.addf %a, %b` -> `pypto.add %a, %b`.
/// The raw (non-adaptor) operands are used on purpose: values already
/// rewritten point at their replacements, and anything else keeps its
/// source type so the converter never materializes casts. During the
/// gradual migration the pypto op keeps the source op's result type;
/// convert-pypto-to-emitpy lowers it into the `pl.*` call_opaque form.
/// Scalar operands keep the arith op: pypto is a tile dialect, and scalar
/// arithmetic rides through to stage two, which maps it onto infix
/// pseudo-callees named after the arith ops.
template <typename SourceOp, typename PyptoOp>
class OpToPyptoOpPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op->getOperands(),
                      [](Value operand) { return isa<VectorType, npuvector::NPUVectorType>(operand.getType()); })) {
      return rewriter.notifyMatchFailure(op, "scalar arithmetic stays arith");
    }
    rewriter.replaceOpWithNewOp<PyptoOp>(op, op->getResultTypes(), op->getOperands());
    return success();
  }
};

/// Rewrites signed min/max into the pypto op chosen by operand kind:
/// scalars go to `pypto.max`/`pypto.min` (PyPTO's scalar-only forms),
/// tiles keep `pypto.maximum`/`pypto.minimum` (elementwise, tile-only
/// forms).
template <typename SourceOp, typename ScalarOp, typename TileOp>
class MinMaxToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType, npuvector::NPUVectorType>(op.getLhs().getType())) {
      rewriter.replaceOpWithNewOp<TileOp>(op, op->getResultTypes(), op->getOperands());
    } else {
      rewriter.replaceOpWithNewOp<ScalarOp>(op, op->getResultTypes(), op->getOperands());
    }
    return success();
  }
};

/// Rewrites an integer remainder into the pypto tile op; scalar operands
/// keep the arith op for stage two (mirroring floordivsi): the runtime's
/// `pl.fmod` is a Tile-only API, so scalar remainders reach the translator
/// as the `remsi`/`remui` pseudo-callees and print as the infix `(a % b)`.
template <typename SourceOp>
class TileRemToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<VectorType, npuvector::NPUVectorType>(op.getLhs().getType())) {
      return rewriter.notifyMatchFailure(op, "scalar remainder waits for stage two");
    }
    rewriter.replaceOpWithNewOp<pypto::FmodOp>(op, op->getResultTypes(), op->getOperands());
    return success();
  }
};

/// `pl.cmp` comparison type codes: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge.
static std::optional<unsigned> cmpTypeCodeFor(arith::CmpIPredicate predicate) {
  switch (predicate) {
    case arith::CmpIPredicate::eq:
      return 0u;
    case arith::CmpIPredicate::ne:
      return 1u;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      return 2u;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      return 3u;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      return 4u;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      return 5u;
  }
  return std::nullopt;
}

static std::optional<unsigned> cmpTypeCodeFor(arith::CmpFPredicate predicate) {
  switch (predicate) {
    case arith::CmpFPredicate::OEQ:
      return 0u;
    case arith::CmpFPredicate::ONE:
    case arith::CmpFPredicate::UNE:
      return 1u;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::ULT:
      return 2u;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::ULE:
      return 3u;
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::UGT:
      return 4u;
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::UGE:
      return 5u;
    default:
      // The remaining NaN-sensitive predicates (UEQ, ONE-with-NaN) have no
      // pl.cmp counterpart: torch-mlir lowers plain `<`/`<=`/`>`/`>=` onto
      // the unordered spellings, which differ from the ordered ones only
      // when an operand is NaN.
      return std::nullopt;
    }
}

/// Finalizes a tile comparison rewrite: `mask` is the freshly built
/// `pypto.cmp` and `replaced` the comparison (or comparison broadcast)
/// being rewritten, whose result uses must now read from the mask. A tile
/// comparison returns the packed predicate mask (one byte per eight
/// lanes), which a tile select consumes directly as its condition. Every
/// other consumer -- casts, reshapes, stores, the arithmetic elementwise
/// ops -- expects an elementwise 0/1 tile, so it reads the mask through
/// an unpacking select `select(mask, full(1), full(0))`; `pl.const`
/// cannot spell a BOOL literal, so both select branches are full tiles. A
/// broadcast is not a value consumer: it only relays the mask to the
/// select's geometry, so its source stays the packed mask. A dynamically
/// shaped result has no extents to bind a tile_full to and keeps the bare
/// mask.
static void replaceWithPackedMask(Operation *replaced, Value mask, ConversionPatternRewriter &rewriter) {
  auto keepsPackedMask = [](OpOperand &use) {
    // A broadcast keeps the mask: its rewrite consumes the comparison
    // itself (BroadcastCmpMaskPattern fuses it, or the plain broadcast
    // materialization lifts the i1 tile), and any elementwise consumer
    // below the broadcast gets its unpacking select at the broadcast's
    // geometry from that rewrite. Retargeting the source here instead
    // would leave the broadcast's i1 result reading an i32 source, which
    // the row/col_expand materialization rejects outright.
    if (isa<npuvector::BroadcastOp, vector::BroadcastOp>(use.getOwner())) {
      return true;
    }
    return use.getOperandNumber() == 0 &&
           isa<arith::SelectOp, npuvector::SelectOp, pypto::SelectOp>(use.getOwner());
  };
  bool needsUnpack = llvm::any_of(replaced->getResult(0).getUses(),
                                  [&](OpOperand &use) { return !keepsPackedMask(use); });
  Type resultType = replaced->getResult(0).getType();
  bool staticShape = false;
  if (auto shaped = dyn_cast<ShapedType>(resultType)) {
    staticShape = shaped.hasStaticShape();
  } else if (auto tile = dyn_cast<pypto::TileType>(resultType)) {
    staticShape = llvm::none_of(tile.getShape(), ShapedType::isDynamic);
  }
  if (needsUnpack && staticShape) {
    Location loc = replaced->getLoc();
    // A2/A3 has no 8-bit form for tsel/texpands, so the unpacking spells
    // its branches in i32 -- `pl.const` cannot spell a BOOL literal either
    // -- and the consumer's cast re-derives its dtype from there (a cast
    // onto i32 itself folds away as a no-op).
    Type elem = rewriter.getI32Type();
    Type unpackType;
    if (auto tile = dyn_cast<pypto::TileType>(resultType)) {
      unpackType = pypto::TileType::get(rewriter.getContext(), elem, tile.getShape());
    } else if (auto nv = dyn_cast<npuvector::NPUVectorType>(resultType)) {
      unpackType = nv.cloneWith(nv.getShape(), elem);
    } else {
      auto shaped = cast<ShapedType>(resultType);
      unpackType = VectorType::get(shaped.getShape(), elem);
    }
    Value ones = rewriter.create<pypto::TileFullOp>(loc, unpackType, rewriter.getStringAttr("1"), ValueRange{})
                   .getResult();
    Value zeros = rewriter.create<pypto::TileFullOp>(loc, unpackType, rewriter.getStringAttr("0"), ValueRange{})
                    .getResult();
    Value unpacked = rewriter.create<pypto::SelectOp>(loc, unpackType, mask, ones, zeros);
    // Collect the elementwise uses before any rebinding: the rewrite below
    // maps the comparison onto the same-typed mask, so the deferred
    // commit-time rebind only ever sees the i1 readers (select conditions
    // and broadcast sources).
    SmallVector<OpOperand *> unpackUses;
    for (OpOperand &use : replaced->getResult(0).getUses()) {
      if (!keepsPackedMask(use))
        unpackUses.push_back(&use);
    }
    // A replacement whose type differs from the original result never
    // rebinds its uses, so route the comparison through the bare mask and
    // retarget the elementwise consumers onto the unpacking select by hand.
    rewriter.replaceOp(replaced, mask);
    for (OpOperand *use : unpackUses) {
      rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(unpacked); });
    }
    return;
  }
  rewriter.replaceOp(replaced, mask);
}

/// Rewrites an integer/float tile comparison into `pypto.cmp`: the
/// predicate maps onto the matching PyPTO comparison kind, and the
/// `pl.cmp` type codes coincide with the enum values (0=eq .. 5=ge).
/// Scalar comparisons keep the arith op: `pl.cmp` has no scalar overload,
/// so stage two maps them onto the infix eq/ne/lt/le/gt/ge pseudo-callees.
template <typename SourceOp>
class CmpToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto code = cmpTypeCodeFor(op.getPredicate());
    if (!code) {
      return rewriter.notifyMatchFailure(op, "comparison predicate has no pl.cmp counterpart");
    }
    if (llvm::none_of(op->getOperands(),
                      [](Value operand) { return isa<VectorType, npuvector::NPUVectorType>(operand.getType()); })) {
      // A scalar arith comparison stays put; a scalar npuvector
      // comparison cannot, so it rewrites onto the arith spelling,
      // which stage two renders with the infix pseudo-callees.
      if constexpr (std::is_same_v<SourceOp, npuvector::CmpFOp>) {
        rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, op.getPredicate(), op.getLhs(), op.getRhs());
        return success();
      }
      if constexpr (std::is_same_v<SourceOp, npuvector::CmpIOp>) {
        rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, op.getPredicate(), op.getLhs(), op.getRhs());
        return success();
      }
      return rewriter.notifyMatchFailure(op, "scalar comparison stays arith");
    }
    auto predicate = static_cast<pypto::CmpPredicate>(*code);
    Location loc = op->getLoc();
    Type resultType = op->getResult(0).getType();
    Value mask =
      rewriter.create<pypto::CmpOp>(loc, resultType, predicate, op->getOperands()[0], op->getOperands()[1]);
    replaceWithPackedMask(op, mask, rewriter);
    return success();
  }
};

/// Rewrites a numeric cast into `pypto.cast`: the target dtype travels in
/// the result type, and convert-pypto-to-emitpy re-derives the
/// `pl.cast(v, pl.<dtype>)` argument from it.
template <typename SourceOp>
class CastToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op, "cast target type has no PyPTO dtype");
    }
    if (op->getOperand(0).getType() == resultType) {
      // A comparison unpacking already materialized this dtype (the mask
      // select spells its branches in i32); ptoas rejects a same-dtype
      // tile cast outright, so the cast folds into a plain forwarding.
      rewriter.replaceOp(op, op->getOperand(0));
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::CastOp>(op, resultType, op->getOperand(0));
    return success();
  }
};

/// Rewrites a bitcast into `pypto.reinterpret_view`: the target dtype
/// travels in the result type, and convert-pypto-to-emitpy re-derives the
/// `pl.reinterpret_view(v, pl.<dtype>)` argument from it.
template <typename SourceOp>
class BitcastToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op, "cast target type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::ReinterpretViewOp>(op, resultType, op->getOperand(0));
    return success();
  }
};

/// Rewrites `arith.constant` into EmitPy constant producers:
/// splat tiles -> `pl.tile.full(shape, dtype, v)` via `pypto.tile_full`,
/// scalars -> `pypto.constant` with the Python literal as text. Scalar
/// literals stay Python values: PyPTO accepts them wherever a scalar operand
/// is expected. Index constants keep their `arith.constant` form here --
/// only convert-pypto-to-emitpy lowers them to `emitpy.literal`.
class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
 public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    TypedAttr value = op.getValue();
    Type type = op.getType();

    // Index constants stay `arith.constant` (see the class comment).
    if (isa<IndexType>(type)) {
      return rewriter.notifyMatchFailure(op, "index constants stay arith.constant for stage two");
    }

    // Splat tile constants map onto `pl.tile.full(shape, dtype, value)`.
    if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
      auto shaped = dyn_cast<ShapedType>(type);
      if (!shaped || !shaped.hasStaticShape() || !dense.isSplat()) {
        return rewriter.notifyMatchFailure(op, "only static splat tile constants map to pl.tile.full");
      }
      auto dtype = pyptoDtypeName(shaped.getElementType());
      if (!dtype) {
        return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
      }
      std::optional<std::string> elem;
      if (isa<FloatType>(shaped.getElementType())) {
        elem = floatToPythonText(dense.getSplatValue<APFloat>());
      } else {
        elem = intToPythonText(dense.getSplatValue<APInt>());
      }
      if (!elem) {
        return rewriter.notifyMatchFailure(op, "unsupported splat value");
      }
      rewriter.replaceOpWithNewOp<pypto::TileFullOp>(op, type, rewriter.getStringAttr(*elem), ValueRange{});
      return success();
    }

    std::optional<std::string> text;
    if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      text = floatToPythonText(floatAttr.getValue());
    } else if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (intAttr.getType().isInteger(1)) {
        text = intAttr.getValue().getBoolValue() ? std::string("True") : std::string("False");
      } else {
        text = intToPythonText(intAttr.getValue());
      }
    }
    if (!text) {
      return rewriter.notifyMatchFailure(op, "unsupported constant value");
    }
    // Scalar constants carry their exact dtype (kept in the result type):
    // a bare Python literal would default to index/float64, and scalar ops
    // such as pl.write require the value dtype to match the tensor dtype.
    auto dtype = pyptoDtypeName(type);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "scalar type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::ConstantOp>(op, type, rewriter.getStringAttr(*text));
    return success();
  }
};

/// Rewrites `transfer_read` into
/// `%r = pl.load(src, [o...], [shape])`; offsets may be loop-dependent.
/// A dynamic `npuvector.transfer_read src[off] [sizes] [max], pad` instead
/// becomes `pl.load(src, [o...], shapes, valid_shape)`: `shapes` takes the
/// per-dimension upper bounds (`maxSizes`) so the tile is allocated at the
/// full extent, and `valid_shape` takes the runtime sizes so a tail tile
/// never reads or stores past the end of the source. A statically shaped
/// read whose row is not a whole number of 32-byte units takes the same
/// padded/valid form, so PTOAS can allocate the tile at all.
template <typename TransferReadOp>
class TransferReadPattern : public OpConversionPattern<TransferReadOp> {
 public:
  using OpConversionPattern<TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TransferReadOp op, typename TransferReadOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getMask()) {
      return rewriter.notifyMatchFailure(op, "masked reads are not supported");
    }
    auto shaped = dyn_cast<ShapedType>(op.getVector().getType());
    if (!shaped) {
      return rewriter.notifyMatchFailure(op, "read result must be a shaped type");
    }
    if constexpr (std::is_same_v<TransferReadOp, npuvector::TransferReadOp>) {
      if (!shaped.hasStaticShape()) {
        return rewriteDynamicRead(op, shaped, rewriter);
      }
    }
    if (!shaped.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "read result must be statically shaped");
    }
    // PTOAS allocates tiles in whole 32-byte rows: a read whose row
    // (last-dim extent times the element size) is not a whole number of
    // 32-byte units cannot be allocated as-is. Such reads take the
    // padded/valid form: the allocation bound rounds up to an aligned
    // row while the valid extent keeps bounding the data actually moved.
    unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t cols = shaped.getDimSize(shaped.getRank() - 1);
    if (elemBytes == 0 || cols <= 0 || (cols * static_cast<int64_t>(elemBytes)) % 32 == 0) {
      // A rank-reduced read (source rank > tile rank) keeps addressing
      // every source dimension: convert-pypto-to-emitpy pads the tile's
      // shape with leading ones so `pl.load` shares the source's
      // coordinate system, and the runtime then returns a tile with the
      // source's rank while downstream ops are typed against the tile's
      // own (physical) rank -- restore the tile's shape with a
      // `pypto.reshape` right after the load.
      auto srcType = dyn_cast<ShapedType>(op.getSource().getType());
      if (srcType && srcType.getRank() > shaped.getRank()) {
        Value load = rewriter.create<pypto::LoadOp>(op.getLoc(), op.getVector().getType(), op.getSource(),
                                                    op.getIndices(), ValueRange{}, ValueRange{});
        rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getVector().getType(), load);
        return success();
      }
      rewriter.replaceOpWithNewOp<pypto::LoadOp>(op, op.getVector().getType(), op.getSource(), op.getIndices(),
                                                 ValueRange{}, ValueRange{});
      return success();
    }
    return rewritePaddedStaticRead(op, shaped, rewriter);
  }

 private:
  /// Maps each tile axis onto the source dimension it reads: a
  /// rank-reduced read projects each tile axis onto one source dimension
  /// through the permutation map, and the dropped source dimensions stay
  /// addressed by the offsets alone. A missing map defaults to the
  /// trailing-dimensions (minor identity) projection. Returns an empty
  /// vector when the map does not name a distinct source dimension per
  /// tile axis.
  static SmallVector<unsigned> tileToSourceDims(TransferReadOp op, unsigned tileRank, unsigned srcRank) {
    if (srcRank == tileRank) {
      SmallVector<unsigned> identity(tileRank);
      std::iota(identity.begin(), identity.end(), 0u);
      return identity;
    }
    SmallVector<unsigned> tileToSrc;
    std::optional<AffineMap> perm = op.getPermutationMap();
    if (perm) {
      if (perm->getNumResults() == tileRank) {
        for (AffineExpr expr : perm->getResults()) {
          auto dim = dyn_cast<AffineDimExpr>(expr);
          if (dim && dim.getPosition() < srcRank && !llvm::is_contained(tileToSrc, dim.getPosition())) {
            tileToSrc.push_back(dim.getPosition());
            continue;
          }
          tileToSrc.clear();
          break;
        }
      }
    } else {
      for (unsigned dim = 0; dim < tileRank; ++dim) {
        tileToSrc.push_back(srcRank - tileRank + dim);
      }
    }
    return tileToSrc;
  }

  /// Statically shaped read whose row is not 32-byte aligned: emit the
  /// padded/valid form. `paddedShapes` names the per-dimension allocation
  /// bounds in the source's coordinate system (one entry per source
  /// dimension): the tile's row extent rounds up to a whole 32-byte unit,
  /// every other axis keeps its extent, and the dropped dimensions of a
  /// rank-reduced read span a single element. `validExtents` carries the
  /// true extents, which bound the data actually moved, so the padding
  /// never reads past the source.
  LogicalResult rewritePaddedStaticRead(TransferReadOp op, ShapedType shaped,
                                        ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned tileRank = shaped.getRank();
    auto sourceType = dyn_cast<ShapedType>(op.getSource().getType());
    unsigned srcRank = sourceType ? sourceType.getRank() : tileRank;
    // Map each tile axis onto the source dimension it reads: a
    // rank-reduced read projects each tile axis onto one source
    // dimension through the permutation map; the dropped source
    // dimensions stay addressed by the offsets alone. A missing map
    // defaults to the trailing-dimensions (minor identity) projection.
    SmallVector<unsigned> tileToSrc = tileToSourceDims(op, tileRank, srcRank);
    if (tileToSrc.size() != tileRank) {
      return rewriter.notifyMatchFailure(
        op, "rank-reduced reads must project each tile axis onto a distinct source dimension");
    }
    SmallVector<Value> paddedValues(srcRank);
    SmallVector<Value> validValues(srcRank);
    if (srcRank > tileRank) {
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      paddedValues.assign(srcRank, one);
      validValues.assign(srcRank, one);
    }
    for (unsigned tileDim = 0; tileDim < tileRank; ++tileDim) {
      int64_t size = shaped.getDimSize(tileDim);
      int64_t bound = tileDim + 1 == tileRank ? alignExtentTo32B(size, elemBytes) : size;
      paddedValues[tileToSrc[tileDim]] = rewriter.create<arith::ConstantIndexOp>(loc, bound);
      validValues[tileToSrc[tileDim]] = rewriter.create<arith::ConstantIndexOp>(loc, size);
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(op, op.getVector().getType(), op.getSource(), op.getIndices(),
                                               ValueRange(paddedValues), ValueRange(validValues));
    return success();
  }

  /// Dynamic-extent read. `dynamicSizes` names the per-dimension extent of
  /// the result tile in one of two arities: one value per dimension (the
  /// vectorizer emits constants for statically sized axes too) or, when
  /// every dynamic slot only is covered, one per `?` in the result shape.
  /// `maxSizes` (when present, same arity as `dynamicSizes`) names the
  /// tile's allocated extent. A rank-reduced read (source rank > tile
  /// rank) drops leading source dimensions: the tile maps onto the
  /// trailing ones (minor identity), so its shape is padded with leading
  /// ones up to the source rank -- offsets still address every source
  /// dimension.
  LogicalResult rewriteDynamicRead(npuvector::TransferReadOp op, ShapedType shaped,
                                   ConversionPatternRewriter &rewriter) const {
    unsigned tileRank = shaped.getRank();
    unsigned numDynamic = llvm::count(shaped.getShape(), ShapedType::kDynamic);
    ValueRange dynamicSizes = op.getDynamicSizes();
    bool perDim = dynamicSizes.size() == tileRank;
    bool perQuestion = dynamicSizes.size() == numDynamic;
    if (!perDim && !perQuestion) {
      return rewriter.notifyMatchFailure(op,
                                         "dynamic sizes must match either the tile's rank or the number "
                                         "of dynamic extents");
    }
    auto sourceType = dyn_cast<ShapedType>(op.getSource().getType());
    unsigned srcRank = sourceType ? sourceType.getRank() : tileRank;
    // A rank-reduced read projects each tile axis onto one source
    // dimension through the permutation map; the dropped source
    // dimensions stay addressed by the offsets alone. A missing map
    // defaults to the trailing-dimensions (minor identity) projection.
    SmallVector<unsigned> tileToSrc;
    if (srcRank > tileRank) {
      tileToSrc = tileToSourceDims(op, tileRank, srcRank);
      if (tileToSrc.size() != tileRank) {
        return rewriter.notifyMatchFailure(
          op, "rank-reduced reads must project each tile axis onto a distinct source dimension");
      }
    }
    Location loc = op.getLoc();
    ValueRange maxSizes = op.getMaxSizes();
    // Per dimension: the tile's allocated extent, and the valid extent.
    SmallVector<Value> shapeValues;
    SmallVector<Value> validValues;
    unsigned dynIdx = 0;
    for (unsigned dim = 0; dim < tileRank; ++dim) {
      int64_t size = shaped.getDimSize(dim);
      if (perDim || size == ShapedType::kDynamic) {
        Value extent = dynamicSizes[perDim ? dim : dynIdx];
        if (size == ShapedType::kDynamic) {
          ++dynIdx;
        }
        unsigned allocIdx = perDim ? dim : dynIdx - 1;
        Value alloc = allocIdx < maxSizes.size() ? maxSizes[allocIdx] : extent;
        // PTOAS allocates whole 32-byte rows: pad a constant bound so the
        // allocation is addressable; the valid extent still bounds the
        // data actually moved.
        if (auto bound = constIndexValue(alloc)) {
          unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
          alloc = rewriter.create<arith::ConstantIndexOp>(loc, alignExtentTo32B(*bound, elemBytes));
        }
        shapeValues.push_back(alloc);
        validValues.push_back(extent);
        continue;
      }
      Value extent = rewriter.create<arith::ConstantIndexOp>(loc, size);
      shapeValues.push_back(extent);
      validValues.push_back(extent);
    }
    // Rank reduction: dropped source dimensions span a single element;
    // the tile's axes carry their extents at the source dimensions the
    // permutation map names.
    if (srcRank > tileRank) {
      SmallVector<Value> srcShape(srcRank);
      SmallVector<Value> srcValid(srcRank);
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (unsigned dim = 0; dim < srcRank; ++dim) {
        srcShape[dim] = one;
        srcValid[dim] = one;
      }
      for (unsigned tileDim = 0; tileDim < tileRank; ++tileDim) {
        srcShape[tileToSrc[tileDim]] = shapeValues[tileDim];
        srcValid[tileToSrc[tileDim]] = validValues[tileDim];
      }
      shapeValues = std::move(srcShape);
      validValues = std::move(srcValid);
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(op, op.getVector().getType(), op.getSource(), op.getIndices(),
                                               ValueRange(shapeValues), ValueRange(validValues));
    return success();
  }
};

/// Rewrites `transfer_write` into `pl.store(v, [o...], dest)`; offsets may
/// be loop-dependent. Device backends impose two extra rules:
/// - tile stores need a 2D tile and a matching-rank tensor, so rank-1
///   values are stored through a `[1, N]` reshape when the row is a whole
///   number of 32-byte units, and element-wise through `pl.write`
///   otherwise;
/// - a value that is just a broadcast scalar is written directly with
///   `pl.write(dest, [o...], scalar)` instead of materializing a tile.
template <typename TransferWriteOp>
class TransferWritePattern : public OpConversionPattern<TransferWriteOp> {
 public:
  using OpConversionPattern<TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TransferWriteOp op, typename TransferWriteOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getMask()) {
      return rewriter.notifyMatchFailure(op, "masked writes are not supported");
    }
    Location loc = op.getLoc();

    // A broadcast scalar feeding only this write is written directly; the
    // broadcast (or its converted add-onto-carrier form) is left dead for
    // the post-conversion cleanup to drop.
    Value vector = op.getVector();
    if (Value scalar = scalarBroadcastSource(vector)) {
      writeScalar(rewriter, loc, op.getSource(), op.getIndices(), scalar);
      finishRewrite(rewriter, op);
      return success();
    }

    // A scalar value into a single-element destination: 0-D and [1] tiles
    // are the shape-normalized forms of scalars, which have no tile form on
    // the device -- write the value directly with pl.write, addressing
    // every destination dimension at 0.
    auto scalarDestType = dyn_cast<ShapedType>(op.getSource().getType());
    if (isScalarToSingleElementDest(vector, scalarDestType)) {
      writeScalarToSingleElement(rewriter, loc, op.getSource(), vector, scalarDestType);
      finishRewrite(rewriter, op);
      return success();
    }

    // A scalar value into a multi-element destination: lowerZeroDimChains
    // lowers whole single-element-tile chains onto plain scalars, so the
    // write forwards the transfer's own coordinates to pl.write.
    if (isa<FloatType, IntegerType, IndexType>(vector.getType())) {
      writeScalar(rewriter, loc, op.getSource(), op.getIndices(), vector);
      finishRewrite(rewriter, op);
      return success();
    }

    auto destType = dyn_cast<ShapedType>(op.getSource().getType());
    auto valueType = dyn_cast<ShapedType>(vector.getType());

    // Rank-1 values go through a [1, N] reshape: tile stores need a 2D
    // tile. The destination is either rank-1 (a row offset of 0 is
    // prepended) or rank-2 (the tile's own row offset is used directly).
    // The value extent may be dynamic; the read that allocated the tile
    // bounds N.
    if (valueType && destType && valueType.getRank() == 1 && destType.hasStaticShape() &&
        (destType.getRank() == 1 || destType.getRank() == 2)) {
      if (failed(storeRankOneViaReshape(rewriter, loc, op, vector, destType, valueType))) {
        return failure();
      }
      finishRewrite(rewriter, op);
      return success();
    }

    // A rank-reduced write (destination rank > tile rank) stores through
    // a padded [1, ..., 1, N] view: the load-side lowering reshaped the
    // tile back to its physical rank right after the read, and the store
    // must re-enter the destination's coordinate system. The padded tile
    // rides through FlattenTileNdTo2D like any other ND tile, whose 2D
    // flattening is what `pl.store` requires -- a bare rank-1 tile is
    // never flattened and fails PTO's 2D valid-shape check. Higher-rank
    // tiles are already 2D-collapsible and keep the default store.
    if (valueType && destType && valueType.getRank() == 1 && destType.getRank() > 2) {
      if (failed(storeRankReducedPadded(rewriter, loc, op, vector, destType, valueType))) {
        return failure();
      }
      finishRewrite(rewriter, op);
      return success();
    }

    rewriter.create<pypto::StoreOp>(loc, op.getVector(), op.getSource(), op.getIndices());
    finishRewrite(rewriter, op);
    return success();
  }

 private:
  /// True when `vector` is a plain scalar value stored into a statically
  /// shaped single-element tensor.
  static bool isScalarToSingleElementDest(Value vector, ShapedType destType) {
    return isa<FloatType, IntegerType, IndexType>(vector.getType()) && destType && destType.hasStaticShape() &&
           destType.getNumElements() == 1;
  }

  /// If `vector` is a scalar broadcast feeding a single user -- either the
  /// still-unconverted broadcast op or its converted `pypto.add(carrier, s)`
  /// form -- returns the broadcast scalar.
  static Value scalarBroadcastSource(Value vector) {
    auto isScalar = [](Type type) { return isa<FloatType>(type) || isa<IntegerType>(type) || isa<IndexType>(type); };
    Operation *def = vector.getDefiningOp();
    if (!def || !vector.hasOneUse()) {
      return Value();
    }
    if (auto bcast = dyn_cast<vector::BroadcastOp>(def)) {
      if (!isScalar(bcast.getSource().getType())) {
        return Value();
      }
      return bcast.getSource();
    }
    if (auto bcast = dyn_cast<npuvector::BroadcastOp>(def)) {
      if (!isScalar(bcast.getSource().getType())) {
        return Value();
      }
      return bcast.getSource();
    }
    if (auto add = dyn_cast<pypto::AddOp>(def)) {
      // The converted runtime-scalar broadcast: pypto.add(carrier, s)
      // with a statically-shaped pypto.tile_full carrier.
      Value scalar = add.getRhs();
      if (!isScalar(scalar.getType())) {
        return Value();
      }
      auto carrier = add.getLhs().getDefiningOp<pypto::TileFullOp>();
      if (!carrier || carrier->getNumOperands() != 0) {
        return Value();
      }
      return scalar;
    }
    return Value();
  }

  /// Reads one scalar element out of a tile: `pl.read(tile, [k])`.
  static Value scalarRead(ConversionPatternRewriter &rewriter, Location loc, Value tile, Type elemType, int64_t index) {
    Value coord = rewriter.create<arith::ConstantIndexOp>(loc, index);
    return rewriter.create<pypto::ReadOp>(loc, elemType, tile, ValueRange(coord)).getResult();
  }

  /// Writes one scalar into `dest` at `indices` with `pl.write`.
  static void writeScalar(ConversionPatternRewriter &rewriter, Location loc, Value dest, ValueRange indices,
                          Value scalar) {
    rewriter.create<pypto::WriteOp>(loc, dest, scalar, indices);
  }

  /// Writes a scalar into a single-element destination with `pl.write`,
  /// addressing every destination dimension at 0.
  static void writeScalarToSingleElement(ConversionPatternRewriter &rewriter, Location loc, Value dest, Value scalar,
                                         ShapedType destType) {
    SmallVector<Value> offsets;
    for (int64_t d = 0; d < destType.getRank(); ++d) {
      offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
    rewriter.create<pypto::WriteOp>(loc, dest, scalar, ValueRange(offsets));
  }

  /// The allocation bound of a dynamic rank-1 tile produced by an
  /// unconverted `transfer_read`: its first `maxSizes` entry, padded to
  /// whole 32-byte rows as the read's rewrite will do.
  static std::optional<int64_t> readAllocationBound(npuvector::TransferReadOp read, Value v) {
    ValueRange maxSizes = read.getMaxSizes();
    if (maxSizes.empty()) {
      return std::nullopt;
    }
    auto bound = constIndexValue(maxSizes[0]);
    if (!bound) {
      return std::nullopt;
    }
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      return alignExtentTo32B(*bound, elemBytes);
    }
    return bound;
  }

  /// The allocation bound carried by a rewritten dynamic read: the
  /// `pypto.load`'s padded-shape operand of the dynamic dimension.
  static std::optional<int64_t> loadAllocationBound(pypto::LoadOp load, Value v) {
    if (load.getPaddedShapes().empty()) {
      return std::nullopt;
    }
    std::optional<int64_t> bound = constIndexValue(load.getPaddedShapes()[0]);
    if (!bound) {
      return std::nullopt;
    }
    // The padded extent already carries the padded allocation when the
    // read was rewritten first; padding again is a no-op then, and
    // repairs an unpadded bound otherwise.
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      return alignExtentTo32B(*bound, elemBytes);
    }
    return bound;
  }

  /// Whether `op` is a converted element-wise tile op that keeps the
  /// producing read's allocation.
  static bool isElementWiseTileOp(Operation *op) {
    return isa<pypto::AddOp, pypto::SubOp, pypto::MulOp, pypto::DivOp, pypto::MaximumOp, pypto::MinimumOp,
                pypto::AbsOp, pypto::ExpOp, pypto::LogOp, pypto::SqrtOp, pypto::RsqrtOp, pypto::SinOp,
                pypto::CosOp>(op);
  }

  /// The allocation bound a broadcast names for its own result: its first
  /// `maxSizes` entry, padded to whole 32-byte rows the way the allocation
  /// will be.
  static std::optional<int64_t> broadcastAllocationBound(npuvector::BroadcastOp broadcast, Value v) {
    ValueRange maxSizes = broadcast.getMaxSizes();
    if (maxSizes.empty()) {
      return std::nullopt;
    }
    auto bound = constIndexValue(maxSizes[0]);
    if (!bound) {
      return std::nullopt;
    }
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      return alignExtentTo32B(*bound, elemBytes);
    }
    return bound;
  }

  /// The unconverted partial reduction: patterns run in arbitrary order, so
  /// a dynamic store may still see this form. The rank-1 result keeps the
  /// input's allocation on the surviving axis, mirroring the pypto form
  /// below.
  static std::optional<int64_t> npuvectorReductionExtent(npuvector::ReductionOp reduction, Value v,
                                                         unsigned depth) {
    auto shaped = dyn_cast<ShapedType>(v.getType());
    auto dims = reduction.getReductionDims();
    if (!shaped || shaped.getRank() != 1 || !dims || dims->size() != 1) {
      return std::nullopt;
    }
    auto bounds = tileAllocationShape(reduction.getVector(), depth + 1);
    if (!bounds || bounds->size() != 2) {
      return std::nullopt;
    }
    return (*dims)[0] == 1 ? (*bounds)[0] : (*bounds)[1];
  }

  /// A partial reduction's rank-1 result keeps the input's allocation on
  /// the surviving axis: row folds the last axis, col the first.
  static std::optional<int64_t> pyptoReductionExtent(pypto::ReductionOp reduction, Value v, unsigned depth) {
    auto bounds = tileAllocationShape(reduction.getVector(), depth + 1);
    auto shaped = dyn_cast<ShapedType>(v.getType());
    if (!bounds || !shaped || shaped.getRank() != 1 || bounds->size() != 2) {
      return std::nullopt;
    }
    return reduction.getAxis() == pypto::ReduceAxis::row ? (*bounds)[0] : (*bounds)[1];
  }

  /// The dynamic extent of the first shaped operand, for ops that keep
  /// their operand's allocation.
  static std::optional<int64_t> operandTileExtent(Operation *owner, unsigned depth) {
    for (Value operand : owner->getOperands()) {
      if (!isa<ShapedType>(operand.getType())) {
        continue;
      }
      if (auto extent = dynamicTileExtent(operand, depth + 1)) {
        return extent;
      }
    }
    return std::nullopt;
  }

  /// Whether `op` keeps its operand's dynamic tile allocation.
  static bool keepsTileAllocation(Operation *op) {
    // Converted element-wise tile ops keep the read's allocation.
    if (isElementWiseTileOp(op)) {
      return true;
    }
    // Converted casts (pl.cast) keep the operand's allocation.
    if (isa<pypto::CastOp>(op)) {
      return true;
    }
    // Unconverted element-wise ops keep the read's allocation.
    if (isa<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp, arith::SubIOp, arith::SubFOp,
            arith::DivSIOp, arith::DivUIOp, arith::DivFOp, arith::MaxSIOp, arith::MinSIOp, math::AbsFOp,
            math::ExpOp, math::LogOp, math::SqrtOp, math::RsqrtOp, math::SinOp, math::CosOp, math::RoundOp,
            math::FloorOp, math::CeilOp>(op)) {
      return true;
    }
    // Unconverted casts (arith and npuvector numeric cast families) keep the
    // operand's allocation too: a rank-1 tile fed through truncf/sitofp/...
    // before a dynamic store still names the read's bound.
    if (isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
            arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, npuvector::ExtFOp, npuvector::TruncFOp,
            npuvector::ExtSIOp, npuvector::ExtUIOp, npuvector::TruncIOp, npuvector::SIToFPOp, npuvector::UIToFPOp,
            npuvector::FPToSIOp, npuvector::FPToUIOp>(op)) {
      return true;
    }
    // Selects and comparisons keep the operands' allocation: a dynamic tile
    // fed through a where/select (relu, masked_fill) or a comparison before
    // a dynamic store still names the read's bound. Both the unconverted
    // and the pypto forms appear because patterns run in arbitrary order.
    if (isa<pypto::SelectOp, pypto::CmpOp, npuvector::SelectOp, arith::SelectOp, npuvector::CmpIOp,
            npuvector::CmpFOp, arith::MaximumFOp, arith::MinimumFOp>(op)) {
      return true;
    }
    return false;
  }

  /// The statically-known allocated extent of a dynamic rank-1 tile: the
  /// tile was allocated by a `transfer_read` whose `maxSizes` bound its `?`
  /// extent, and element-wise users keep that allocation. Both the
  /// unconverted read and its `pypto.load` form are handled because patterns
  /// run in arbitrary order.
  static std::optional<int64_t> dynamicTileExtent(Value v, unsigned depth = 0) {
    if (depth > 8) {
      return std::nullopt;
    }
    // Loop-carried tiles: an scf.for result keeps the allocation of the
    // yielded value, and an iter_arg keeps the init's.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (arg.getArgNumber() > 0) {
          return dynamicTileExtent(loop.getInitArgs()[arg.getArgNumber() - 1], depth + 1);
        }
      }
      return std::nullopt;
    }
    Operation *def = v.getDefiningOp();
    if (!def) {
      return std::nullopt;
    }
    if (auto loop = dyn_cast<scf::ForOp>(def)) {
      if (auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator())) {
        for (auto [index, result] : llvm::enumerate(loop.getResults())) {
          if (result == v) {
            return dynamicTileExtent(yield.getOperands()[index], depth + 1);
          }
        }
      }
      return std::nullopt;
    }
    if (auto read = dyn_cast<npuvector::TransferReadOp>(def)) {
      return readAllocationBound(read, v);
    }
    if (auto load = dyn_cast<pypto::LoadOp>(def)) {
      return loadAllocationBound(load, v);
    }
    if (auto broadcast = dyn_cast<npuvector::BroadcastOp>(def)) {
      return broadcastAllocationBound(broadcast, v);
    }
    if (auto reduction = dyn_cast<npuvector::ReductionOp>(def)) {
      return npuvectorReductionExtent(reduction, v, depth);
    }
    if (auto reduction = dyn_cast<pypto::ReductionOp>(def)) {
      return pyptoReductionExtent(reduction, v, depth);
    }
    if (keepsTileAllocation(def)) {
      return operandTileExtent(def, depth);
    }
    return std::nullopt;
  }

  /// Stores a rank-1 value through a `[1, N]` reshape: tile stores need a
  /// 2D tile. The destination is either rank-1 (a row offset of 0 is
  /// prepended) or rank-2 (the tile's own row offset is used directly).
  /// A dynamic extent resolves to the tile's allocated bound; the tile's
  /// valid shape still limits what is written, so tail iterations stay in
  /// bounds. Statically-sized sub-32-byte rows cannot hold a tile and fall
  /// back to element-wise `pl.write`s, which require a rank-1 destination
  /// and a constant base offset.
  static LogicalResult storeRankOneViaReshape(ConversionPatternRewriter &rewriter, Location loc, TransferWriteOp op,
                                              Value vector, ShapedType destType, ShapedType valueType) {
    Type elemType = valueType.getElementType();
    // A dynamic extent names no element count of its own: the tile was
    // allocated by the producing read, whose maxSizes bound it.
    std::optional<int64_t> dynamicExtent;
    int64_t n = valueType.getNumElements();
    if (n == ShapedType::kDynamic) {
      dynamicExtent = dynamicTileExtent(vector);
      if (!dynamicExtent) {
        return rewriter.notifyMatchFailure(op, "dynamic rank-1 store needs the tile's allocated extent");
      }
      n = *dynamicExtent;
    }
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (!dynamicExtent && n * elemBytes % 32 != 0) {
      // Sub-32B rows cannot hold a tile at all: fall back to
      // element-wise scalar writes. The rank-1 value spans the
      // destination's trailing axis, so the leading offsets pass
      // through and only the last one advances per element. `pl.write`
      // accepts dynamic offsets, so the base need not be constant.
      if (std::optional<AffineMap> perm = op.getPermutationMap();
          perm && !perm->isMinorIdentity()) {
        return rewriter.notifyMatchFailure(op, "sub-32B element-wise fallback needs a minor-identity permutation");
      }
      ValueRange indices = op.getIndices();
      SmallVector<Value> leading = llvm::to_vector(indices.drop_back());
      Value base = indices.back();
      for (int64_t k = 0; k < n; ++k) {
        Value elem = scalarRead(rewriter, loc, vector, elemType, k);
        Value offset;
        if (std::optional<int64_t> baseConst = constIndexValue(base)) {
          offset = rewriter.create<arith::ConstantIndexOp>(loc, *baseConst + k);
        } else if (k == 0) {
          offset = base;
        } else {
          offset = rewriter.create<arith::AddIOp>(loc, base, rewriter.create<arith::ConstantIndexOp>(loc, k));
        }
        SmallVector<Value> elemIndices = leading;
        elemIndices.push_back(offset);
        writeScalar(rewriter, loc, op.getSource(), elemIndices, elem);
      }
      return success();
    }
    Value reshaped = reshapeTileOp(rewriter, loc, vector, 1, n, elemType);
    SmallVector<Value> offsetItems;
    Value storeDest = op.getSource();
    if (destType.getRank() == 1) {
      // Prepend the [1, N] row offset, and store through a [1, N] view of
      // the flat destination: PTOAS matches both the offset count and the
      // partition shape to the destination's rank, so a 2D tile needs a
      // 2D destination view.
      offsetItems.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
      Type viewType;
      if (auto memrefType = dyn_cast<MemRefType>(destType)) {
        viewType = MemRefType::get({1, memrefType.getNumElements()}, memrefType.getElementType());
      } else {
        auto tensorType = cast<RankedTensorType>(destType);
        viewType = RankedTensorType::get({1, tensorType.getNumElements()}, tensorType.getElementType());
      }
      // The view is a tensor-level reshape: create it at function scope,
      // next to the destination's own view chain, not beside the store.
      // The PyPTO emitter hoists entry-level view ops above the device
      // scope; created inside it, the reshape would be materialized as a
      // tile op, which tile.store rejects as its destination.
      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *destDef = op.getSource().getDefiningOp()) {
        rewriter.setInsertionPointAfter(destDef);
      } else if (auto funcOp = op->template getParentOfType<func::FuncOp>()) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      }
      storeDest = rewriter.create<pypto::ReshapeOp>(loc, viewType, op.getSource()).getResult();
    }
    llvm::append_range(offsetItems, op.getIndices());
    // `pypto.store` keeps the offsets as explicit index operands;
    // convert-pypto-to-emitpy folds the constant ones back into the
    // `pl.store` offsets list.
    rewriter.create<pypto::StoreOp>(loc, reshaped, storeDest, ValueRange(offsetItems));
    return success();
  }

  /// Stores a rank-reduced value (destination rank > tile rank) through a
  /// `[1, ..., 1, N...]` padded reshape -- the mirror image of the reshape
  /// the load-side lowering inserts after a rank-reduced read. `pl.store`
  /// offsets and shapes share the destination's coordinate system, and
  /// PTO's tile flattening only covers ND tiles: a bare rank-1 tile fails
  /// the 2D valid-shape check, so the value is padded back to the
  /// destination's rank with leading singleton dims. A dynamic extent
  /// resolves to the tile's allocated bound.
  static LogicalResult storeRankReducedPadded(ConversionPatternRewriter &rewriter, Location loc, TransferWriteOp op,
                                              Value vector, ShapedType destType, ShapedType valueType) {
    // The padding assumes the tile spans the destination's trailing
    // dimensions, which is exactly the minor-identity projection.
    if (std::optional<AffineMap> perm = op.getPermutationMap(); perm && !perm->isMinorIdentity()) {
      return rewriter.notifyMatchFailure(op, "rank-reduced padded store needs a minor-identity permutation");
    }
    Type elemType = valueType.getElementType();
    if (!pyptoDtypeName(elemType)) {
      return rewriter.notifyMatchFailure(op, "rank-reduced store element type has no PyPTO dtype");
    }
    SmallVector<int64_t> shape = llvm::to_vector(valueType.getShape());
    if (llvm::any_of(shape, [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      if (shape.size() != 1) {
        return rewriter.notifyMatchFailure(op, "multi-dim dynamic rank-reduced store is not supported");
      }
      std::optional<int64_t> dynamicExtent = dynamicTileExtent(vector);
      if (!dynamicExtent) {
        return rewriter.notifyMatchFailure(op, "dynamic rank-reduced store needs the tile's allocated extent");
      }
      shape[0] = *dynamicExtent;
    }
    SmallVector<int64_t> padded(destType.getRank() - shape.size(), 1);
    llvm::append_range(padded, shape);
    Type paddedType = pypto::TileType::get(rewriter.getContext(), elemType, padded);
    Value reshaped = rewriter.create<pypto::ReshapeOp>(loc, paddedType, vector);
    rewriter.create<pypto::StoreOp>(loc, reshaped, op.getSource(), op.getIndices());
    return success();
  }

  /// Removes the rewritten write.
  static void finishRewrite(ConversionPatternRewriter &rewriter, TransferWriteOp op) {
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      // Result-ful writes alias the destination buffer.
      rewriter.replaceOp(op, op.getSource());
    }
  }
};

/// Reduction kinds map onto the pypto reduction families; bitwise
/// combining has no PyPTO counterpart.
static std::optional<pypto::ReduceKind> reduceKind(vector::CombiningKind kind) {
  switch (kind) {
    case vector::CombiningKind::ADD:
      return pypto::ReduceKind::sum;
    case vector::CombiningKind::MUL:
      return pypto::ReduceKind::prod;
    case vector::CombiningKind::MINUI:
    case vector::CombiningKind::MINSI:
    case vector::CombiningKind::MINNUMF:
    case vector::CombiningKind::MINIMUMF:
      return pypto::ReduceKind::min;
    case vector::CombiningKind::MAXUI:
    case vector::CombiningKind::MAXSI:
    case vector::CombiningKind::MAXNUMF:
    case vector::CombiningKind::MAXIMUMF:
      return pypto::ReduceKind::max;
    default:
      return std::nullopt;
  }
}

/// Whether the combining function has a scalar-form PyPTO op. All four
/// reduction families have scalar forms (`pl.min`/`pl.max` for min/max).
static bool hasScalarCombine(pypto::ReduceKind kind) {
  return true;
}

/// Reads one element out of a tile.
static Value readTileElement(ConversionPatternRewriter &rewriter, Location loc, Type resultType, Value tile,
                             ArrayRef<int64_t> index) {
  SmallVector<Value> coords;
  coords.reserve(index.size());
  llvm::transform(index, std::back_inserter(coords),
                  [&](int64_t k) { return rewriter.create<arith::ConstantIndexOp>(loc, k); });
  return rewriter.create<pypto::ReadOp>(loc, resultType, tile, ValueRange(coords)).getResult();
}

/// Combines two reduction partials with scalar arith; a null `acc`
/// (the first element) passes through unchanged.
static Value combineScalars(ConversionPatternRewriter &rewriter, Location loc, Type resultType, pypto::ReduceKind kind,
                            Value acc, Value elem) {
  if (!acc) {
    return elem;
  }
  // Sum/prod have no type-polymorphic pypto form; spell the integer
  // variants for integer elements.
  bool isFloat = isa<FloatType>(resultType);
  if (kind == pypto::ReduceKind::prod) {
    return isFloat ? rewriter.create<arith::MulFOp>(loc, resultType, acc, elem).getResult()
                   : rewriter.create<arith::MulIOp>(loc, resultType, acc, elem).getResult();
  }
  if (kind == pypto::ReduceKind::min) {
    return rewriter.create<pypto::MinOp>(loc, resultType, acc, elem);
  }
  if (kind == pypto::ReduceKind::max) {
    return rewriter.create<pypto::MaxOp>(loc, resultType, acc, elem);
  }
  return isFloat ? rewriter.create<arith::AddFOp>(loc, resultType, acc, elem).getResult()
                 : rewriter.create<arith::AddIOp>(loc, resultType, acc, elem).getResult();
}

/// Renders flat element index `k` of a col-major tile with extents `dims`
/// as an index list, e.g. [1, 2].
static SmallVector<int64_t> flatIndexToList(ArrayRef<int64_t> dims, int64_t k) {
  SmallVector<int64_t> index;
  for (unsigned d = 0; d < dims.size(); ++d) {
    int64_t stride = 1;
    for (unsigned t = d + 1; t < dims.size(); ++t) {
      stride *= dims[t];
    }
    index.push_back((k / stride) % dims[d]);
  }
  return index;
}

/// Rewrites reductions onto the PyPTO reduction families:
/// - npuvector partial reductions reduce the last axis (`pl.row_*`) or the
///   first axis (`pl.col_*`) and consume a scratch tile;
/// - full reductions (and `vector.reduction`) fold to a scalar through
///   row_* then col_*, reading out the single surviving element. Rank-1
///   inputs are reshaped to [1, N] first because pl.row_* requires 2D.
template <typename ReductionOp>
class ReductionPattern : public OpConversionPattern<ReductionOp> {
 public:
  using OpConversionPattern<ReductionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    std::optional<pypto::ReduceKind> kind = reduceKind(op.getKind());
    if (!kind) {
      return rewriter.notifyMatchFailure(op, "reduction kind has no PyPTO counterpart");
    }

    auto shaped = dyn_cast<ShapedType>(op.getVector().getType());
    if (!shaped || shaped.getRank() == 0) {
      return rewriter.notifyMatchFailure(op, "reduction input must be a ranked tile");
    }
    if (!pyptoDtypeName(shaped.getElementType())) {
      return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
    }

    // Partial reduction: a single reduced axis that must be the first or the
    // last one. Rank-1 reductions always sweep the only axis and fall
    // through to the full-reduction lowering below.
    if constexpr (std::is_same_v<ReductionOp, npuvector::ReductionOp>) {
      auto dims = op.getReductionDims();
      if (dims && !dims->empty() && shaped.getRank() >= 2) {
        return rewritePartialReduction(op, rewriter, *kind, shaped);
      }
    }

    return rewriteFullReduction(op, rewriter, *kind, shaped);
  }

 private:
  /// Reduces one axis in place: the last axis (`row`) or the first axis
  /// (`col`), consuming a scratch tile and folding the accumulator with
  /// an elementwise add.
  static LogicalResult rewritePartialReduction(npuvector::ReductionOp op, ConversionPatternRewriter &rewriter,
                                               pypto::ReduceKind kind, ShapedType shaped) {
    Location loc = op.getLoc();
    auto dims = op.getReductionDims();
    if (dims->size() != 1) {
      return rewriter.notifyMatchFailure(op, "only single-axis reductions are supported");
    }
    int64_t dim = (*dims)[0];
    unsigned rank = shaped.getRank();
    pypto::ReduceAxis axis;
    if (dim == static_cast<int64_t>(rank - 1)) {
      axis = pypto::ReduceAxis::row;
    } else if (dim == 0) {
      axis = pypto::ReduceAxis::col;
    } else {
      return rewriter.notifyMatchFailure(op, "only first/last axis reductions map onto pl.row_*/pl.col_*");
    }
    Type resultType = op->getResult(0).getType();
    SmallVector<int64_t> scratchShape;
    if (shaped.hasStaticShape()) {
      scratchShape.assign(shaped.getShape().begin(), shaped.getShape().end());
    } else {
      // A dynamic dim names no element count of its own: the scratch must
      // cover the tile's allocation, which the producing read/broadcast's
      // maxSizes bound (padded to whole 32-byte rows).
      auto bounds = tileAllocationShape(op.getVector());
      if (!bounds) {
        return rewriter.notifyMatchFailure(op, "a dynamic reduction needs the tile's allocation bounds");
      }
      scratchShape = std::move(*bounds);
    }
    Value tmp = createTmpTile(rewriter, loc, scratchShape, shaped.getElementType());
    Value reduced = rewriter.create<pypto::ReductionOp>(loc, resultType, kind, axis, op.getVector(), tmp).getResult();
    if (op.getAcc()) {
      reduced = rewriter.create<pypto::AddOp>(loc, resultType, reduced, op.getAcc());
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }

  /// Folds a full reduction to a scalar. Device backends cannot allocate
  /// [1, 1] tiles, so the final value is assembled from scalar reads: when
  /// the operand tiles are allocatable (extents a whole number of 32-byte
  /// units) the last axis is reduced on device (`pl.row_*`) and only the
  /// surviving elements are read out; otherwise every element is read
  /// directly. The elements are combined with scalar-form ops --
  /// `pl.minimum`/`pl.maximum` have no scalar form, so min/max full
  /// reductions are rejected.
  static LogicalResult rewriteFullReduction(ReductionOp op, ConversionPatternRewriter &rewriter, pypto::ReduceKind kind,
                                            ShapedType shaped) {
    Location loc = op.getLoc();
    if (!shaped.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "full reductions need a statically shaped tile");
    }
    if (!hasScalarCombine(kind)) {
      return rewriter.notifyMatchFailure(op, "full min/max reductions have no scalar-form PyPTO lowering");
    }
    if (op.getAcc()) {
      return rewriter.notifyMatchFailure(op, "accumulator chaining is not supported for full reductions");
    }

    Type resultType = op->getResult(0).getType();
    Type elemType = shaped.getElementType();
    Value input = op.getVector();
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    // The device walks a col-major tile's contiguous (row) axis in 32-byte
    // steps, so the smallest allocatable row extent is the quantum below
    // (8 rows for a 4-byte dtype).
    unsigned alignRows = 32 / std::gcd(32u, elemBytes);
    int64_t rows = shaped.getRank() == 2 ? shaped.getDimSize(0) : 0;
    int64_t cols = shaped.getRank() == 2 ? shaped.getDimSize(1) : 0;

    Value acc;
    if (shaped.getRank() == 1 && shaped.getNumElements() > 1 && shaped.getNumElements() % alignRows == 0 &&
        (shaped.getNumElements() / alignRows) * elemBytes % 32 == 0) {
      acc = reduceRankOneViaReshape(rewriter, loc, kind, shaped, resultType, input, alignRows);
    } else if (shaped.getRank() == 2 && cols > 1 && rows * elemBytes % 32 == 0 && cols * elemBytes % 32 == 0) {
      acc = reduceViaDeviceRowReduce(rewriter, loc, kind, shaped, resultType, input, rows);
    } else {
      acc = reduceViaScalarChain(rewriter, loc, kind, shaped, resultType, input);
    }
    rewriter.replaceOp(op, acc);
    return success();
  }

  /// Rank-1 lowering: view the tile as [alignRows, N/alignRows] so the
  /// device reduces the last axis into an allocatable [alignRows, 1] tile
  /// (a [1, N] view would reduce to [1, 1], which the device cannot
  /// allocate), then read out and combine the surviving row partials.
  static Value reduceRankOneViaReshape(ConversionPatternRewriter &rewriter, Location loc, pypto::ReduceKind kind,
                                       ShapedType shaped, Type resultType, Value input, unsigned alignRows) {
    int64_t r = alignRows;
    int64_t c = shaped.getNumElements() / r;
    Value reshaped = reshapeTileOp(rewriter, loc, input, r, c, shaped.getElementType());
    Value rowTmp = createTmpTile(rewriter, loc, SmallVector<int64_t>{r, c}, shaped.getElementType());
    Value row =
      rewriter.create<pypto::ReductionOp>(loc, resultType, kind, pypto::ReduceAxis::row, reshaped, rowTmp).getResult();
    return combineRowPartials(rewriter, loc, kind, resultType, row, r);
  }

  /// Hybrid lowering: row-reduce to [rows, 1] on device, then combine the
  /// surviving `rows` scalars.
  static Value reduceViaDeviceRowReduce(ConversionPatternRewriter &rewriter, Location loc, pypto::ReduceKind kind,
                                        ShapedType shaped, Type resultType, Value input, int64_t rows) {
    Value rowTmp = createTmpTile(rewriter, loc, shaped.getShape(), shaped.getElementType());
    Value row =
      rewriter.create<pypto::ReductionOp>(loc, resultType, kind, pypto::ReduceAxis::row, input, rowTmp).getResult();
    return combineRowPartials(rewriter, loc, kind, resultType, row, rows);
  }

  /// Reads the surviving `[i, 0]` partials out of a keepdim row reduction
  /// and combines them with the scalar-form op.
  static Value combineRowPartials(ConversionPatternRewriter &rewriter, Location loc, pypto::ReduceKind kind,
                                  Type resultType, Value row, int64_t rows) {
    Value acc;
    for (int64_t i = 0; i < rows; ++i) {
      Value elem = readTileElement(rewriter, loc, resultType, row, SmallVector<int64_t>{i, 0});
      acc = combineScalars(rewriter, loc, resultType, kind, acc, elem);
    }
    return acc;
  }

  /// Pure scalar chain: read and combine every element.
  static Value reduceViaScalarChain(ConversionPatternRewriter &rewriter, Location loc, pypto::ReduceKind kind,
                                    ShapedType shaped, Type resultType, Value input) {
    SmallVector<int64_t> dims(shaped.getShape());
    Value acc;
    for (int64_t k = 0, e = shaped.getNumElements(); k < e; ++k) {
      Value elem = readTileElement(rewriter, loc, resultType, input, flatIndexToList(dims, k));
      acc = combineScalars(rewriter, loc, resultType, kind, acc, elem);
    }
    return acc;
  }
};

template <typename BroadcastOp>
static FailureOr<SmallVector<Value>> buildBroadcastExtents(BroadcastOp op, ShapedType resultType, Location loc,
                                                           ConversionPatternRewriter &rewriter) {
  if (resultType.hasStaticShape()) {
    return SmallVector<Value>{};
  }
  if constexpr (std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
    ValueRange dynamicSizes = op.getDynamicSizes();
    ValueRange maxSizes = op.getMaxSizes();
    unsigned rank = resultType.getShape().size();
    unsigned numDynamic = llvm::count(resultType.getShape(), ShapedType::kDynamic);
    // `dynamicSizes` follows one of two conventions: per-dynamic-? (one
    // value per `?` extent) or per-tile-dimension (one value per dim, as
    // the linalg vectorization stamps them). An all-dynamic result spells
    // both identically, and keeps the per-dynamic-? reading.
    bool perDim = dynamicSizes.size() == rank && rank != numDynamic;
    if (!perDim && dynamicSizes.size() != numDynamic) {
      return rewriter.notifyMatchFailure(op, "dynamic sizes must match the broadcast's dynamic extents");
    }
    SmallVector<Value> shapeValues;
    unsigned dynIdx = 0;
    for (auto [axis, dim] : llvm::enumerate(resultType.getShape())) {
      if (dim == ShapedType::kDynamic) {
        Value alloc = perDim ? (axis < maxSizes.size() ? maxSizes[axis] : dynamicSizes[axis])
                             : (dynIdx < maxSizes.size() ? maxSizes[dynIdx] : dynamicSizes[dynIdx]);
        // Pad a constant bound to whole 32-byte rows, matching the extent
        // a dynamic transfer_read allocates, so broadcast tiles stay
        // shape-compatible with the loads they combine with.
        if (auto bound = constIndexValue(alloc)) {
          unsigned elemBytes = resultType.getElementType().getIntOrFloatBitWidth() / 8;
          alloc = rewriter.create<arith::ConstantIndexOp>(loc, alignExtentTo32B(*bound, elemBytes));
        }
        shapeValues.push_back(alloc);
        ++dynIdx;
      } else {
        shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }
    }
    return shapeValues;
  }
  return rewriter.notifyMatchFailure(op, "only statically-shaped broadcasts are supported");
}

/// An identity-mapped axis must agree on extent and staticality, or a
/// constant source could be reshaped past its allocation bound.
static bool extentsCompatible(int64_t srcExtent, int64_t resultExtent) {
  return ShapedType::isDynamic(srcExtent) == ShapedType::isDynamic(resultExtent) &&
         (ShapedType::isDynamic(srcExtent) || srcExtent == resultExtent);
}

/// Splits the result axes of a general lift into identity-mapped axes
/// (consumed in increasing order per `dimension`) and dropped axes; the
/// identity-mapped extents must be compatible.
static LogicalResult classifyDroppedAxes(npuvector::NPUVectorType srcType, ShapedType resultType,
                                         ArrayRef<int64_t> dimension, SmallVectorImpl<bool> &isDropped,
                                         SmallVectorImpl<unsigned> &droppedAxes) {
  unsigned rank = resultType.getRank();
  isDropped.assign(rank, false);
  unsigned sourceDim = 0;
  for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
    if (sourceDim < dimension.size() && dimension[sourceDim] == static_cast<int64_t>(resultDim)) {
      if (!extentsCompatible(srcType.getDimSize(sourceDim), resultType.getDimSize(resultDim))) {
        return failure();
      }
      ++sourceDim;
    } else {
      isDropped[resultDim] = true;
      droppedAxes.push_back(resultDim);
    }
  }
  if (sourceDim != dimension.size() || droppedAxes.empty()) {
    return failure();
  }
  return success();
}

/// Lifts one dropped axis of a general lift. The expand family only
/// broadcasts the last two dimensions, so a dropped axis outside them is
/// ferried to the tail with `pypto.transpose` swaps, lifted with a row
/// expand against a zero carrier, and ferried back.
static Value liftDroppedAxis(ShapedType shapedResult, Location loc, StringRef zero, unsigned rank,
                             unsigned dropped, Value cur, const SmallVector<int64_t> &curShape,
                             const SmallVector<Value> &curBounds, const SmallVector<int64_t> &targetShape,
                             const SmallVector<Value> &targetBounds, ConversionPatternRewriter &rewriter) {
  auto swapAxes = [&](SmallVector<int64_t> &shape, SmallVector<Value> &bounds, Value &value, unsigned a,
                      unsigned b) {
    std::swap(shape[a], shape[b]);
    if (!bounds.empty()) {
      std::swap(bounds[a], bounds[b]);
    }
    value = rewriter.create<pypto::TransposeOp>(loc, shapedResult.clone(shape), value,
                                                static_cast<uint64_t>(a), static_cast<uint64_t>(b));
  };
  if (dropped + 2 >= rank) {
    // Tail position (last or second-to-last axis): the expand
    // family broadcasts it directly against a zero carrier.
    Value carrier = rewriter.create<pypto::TileFullOp>(loc, shapedResult.clone(targetShape),
                                                       rewriter.getStringAttr(zero), ValueRange(targetBounds))
                      .getResult();
    return dropped + 1 == rank
             ? rewriter.create<pypto::RowExpandAddOp>(loc, shapedResult.clone(targetShape), carrier, cur).getResult()
             : rewriter.create<pypto::ColExpandAddOp>(loc, shapedResult.clone(targetShape), carrier, cur)
                 .getResult();
  }
  // Ferry the dropped axis to the tail, row-expand there, and
  // ferry back.
  SmallVector<int64_t> vecShape = curShape;
  SmallVector<Value> vecBounds = curBounds;
  Value vec = cur;
  for (unsigned axis = dropped; axis + 1 < rank; ++axis) {
    swapAxes(vecShape, vecBounds, vec, axis, axis + 1);
  }
  SmallVector<int64_t> carrierShape = targetShape;
  SmallVector<Value> carrierBounds = targetBounds;
  for (unsigned axis = dropped; axis + 1 < rank; ++axis) {
    std::swap(carrierShape[axis], carrierShape[axis + 1]);
    if (!carrierBounds.empty()) {
      std::swap(carrierBounds[axis], carrierBounds[axis + 1]);
    }
  }
  Value carrier = rewriter.create<pypto::TileFullOp>(loc, shapedResult.clone(carrierShape),
                                                     rewriter.getStringAttr(zero), ValueRange(carrierBounds))
                    .getResult();
  Value expanded =
    rewriter.create<pypto::RowExpandAddOp>(loc, shapedResult.clone(carrierShape), carrier, vec).getResult();
  for (unsigned axis = rank - 1; axis-- > dropped;) {
    swapAxes(carrierShape, carrierBounds, expanded, axis, axis + 1);
  }
  return expanded;
}

/// General lift onto any rank: `dimension` maps every source axis
/// onto an identity-matched result axis (in increasing order) and
/// every remaining axis is dropped — the lift may span any rank
/// jump and drop axes anywhere. The expand family only broadcasts
/// the last two dimensions, so a dropped axis outside them is
/// ferried to the tail with `pypto.transpose` swaps, lifted with a
/// row expand against a zero carrier, and ferried back.
static LogicalResult materializeGeneralLift(npuvector::BroadcastOp op, ShapedType resultType,
                                            npuvector::NPUVectorType srcType, ArrayRef<int64_t> dimension,
                                            ConversionPatternRewriter &rewriter) {
  unsigned srcRank = srcType.getRank();
  if (dimension.size() != srcRank || resultType.getRank() <= srcRank || resultType.getRank() < 2) {
    return failure();
  }
  unsigned rank = resultType.getRank();
  SmallVector<bool> isDropped;
  SmallVector<unsigned> droppedAxes;
  if (failed(classifyDroppedAxes(srcType, resultType, dimension, isDropped, droppedAxes))) {
    return failure();
  }
  Location loc = op.getLoc();
  Type convertedResult = op->getResult(0).getType();
  FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
  if (failed(extents)) {
    return failure();
  }
  auto shapedResult = cast<ShapedType>(convertedResult);
  auto constant = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
  // Per-dimension runtime bounds of a working shape, one value per
  // axis; an all-static shape carries none, mirroring the pypto
  // ops' empty-extents convention.
  auto makeBounds = [&](const SmallVector<int64_t> &shape) -> SmallVector<Value> {
    if (llvm::none_of(shape, [](int64_t d) { return ShapedType::isDynamic(d); })) {
      return {};
    }
    SmallVector<Value> bounds;
    for (auto [axis, dim] : llvm::enumerate(shape)) {
      bounds.push_back(ShapedType::isDynamic(dim) ? (*extents)[axis] : constant(dim));
    }
    return bounds;
  };
  // Start from the source reshaped with a singleton at every dropped
  // axis; identity axes carry the source extents over.
  SmallVector<int64_t> curShape;
  unsigned sourceDim = 0;
  for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
    curShape.push_back(isDropped[resultDim] ? 1 : srcType.getDimSize(sourceDim++));
  }
  SmallVector<Value> curBounds = makeBounds(curShape);
  Value cur =
    rewriter.create<pypto::ReshapeOp>(loc, shapedResult.clone(curShape), op.getSource(), ValueRange(curBounds));
  std::string zero = isa<FloatType>(resultType.getElementType()) ? "0.0" : "0";
  // Lift one dropped axis at a time, tail first so every expand only
  // meets already-lifted axes behind the one it broadcasts.
  for (auto droppedIt = droppedAxes.rbegin(); droppedIt != droppedAxes.rend(); ++droppedIt) {
    unsigned dropped = *droppedIt;
    SmallVector<int64_t> targetShape = curShape;
    targetShape[dropped] = resultType.getDimSize(dropped);
    SmallVector<Value> targetBounds = makeBounds(targetShape);
    cur = liftDroppedAxis(shapedResult, loc, zero, rank, dropped, cur, curShape, curBounds, targetShape,
                          targetBounds, rewriter);
    curShape = std::move(targetShape);
    curBounds = std::move(targetBounds);
  }
  rewriter.replaceOp(op, cur);
  return success();
}

/// Materializes a rank-1 tile broadcast lifted onto a rank-2 result:
/// `pl.row/col_expand_add(tile_full(0), reshape(src))` spells
/// `0 + broadcast(src)`, a pure promotion that cannot change the lifted
/// values — the tile-to-tile twin of the runtime-scalar zero-carrier
/// trick in the fallback below. Serves the broadcasts every fusion
/// declined: a select value operand, a transfer_write, an integer
/// arith consumer, or a sibling broadcast that already rewrote the
/// shared consumer into the expand family.
///
/// The reshaped vector's dynamic extent mirrors the carrier's (the
/// aligned `maxSizes` bound `buildBroadcastExtents` produces) so the
/// expand combines same-shaped tiles; the identity-mapped axis must
/// agree on extent and staticality, or a constant source could be
/// reshaped past its allocation bound.
static LogicalResult materializeLiftedBroadcast(npuvector::BroadcastOp op, ShapedType resultType,
                                                ConversionPatternRewriter &rewriter) {
  ArrayRef<int64_t> dimension = op.getDimension();
  bool rowLift = dimension.size() == 1 && dimension[0] == 0;
  bool colLift = dimension.size() == 1 && dimension[0] == 1;
  auto srcType = dyn_cast<npuvector::NPUVectorType>(op.getSource().getType());
  if (!srcType) {
    return failure();
  }
  // The row/col spelling below only covers rank-2 promotions of a
  // rank-1 source; a [0]/[1] lift onto a higher-rank result falls
  // through to the general lift, which spells the same promotion
  // (identity-mapped axis + dropped axes) with transposes.
  bool specializedLift = (rowLift || colLift) && resultType.getRank() == 2 && srcType.getRank() == 1;
  if (!specializedLift) {
    return materializeGeneralLift(op, resultType, srcType, dimension, rewriter);
  }

  if (resultType.getRank() != 2) {
    return failure();
  }
  if (srcType.getRank() != 1) {
    return failure();
  }
  // The lifted axis is the identity mapping, so the source and result
  // extents must match — on staticality too.
  unsigned axis = rowLift ? 0 : 1;
  int64_t srcExtent = srcType.getDimSize(0);
  if (!extentsCompatible(srcExtent, resultType.getDimSize(axis))) {
    return failure();
  }
  Location loc = op.getLoc();
  Type convertedResult = op->getResult(0).getType();
  FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
  if (failed(extents)) {
    return failure();
  }
  // The reshaped vector: [rows, 1] for a row lift, [1, cols] for a
  // column lift; a dynamic extent rides the carrier's.
  Type vecType;
  SmallVector<Value> vecExtents;
  if (ShapedType::isDynamic(srcExtent)) {
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    if (rowLift) {
      vecType = srcType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
      vecExtents.push_back((*extents)[0]);
      vecExtents.push_back(one);
    } else {
      vecType = srcType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
      vecExtents.push_back(one);
      vecExtents.push_back((*extents)[1]);
    }
  } else if (rowLift) {
    vecType = srcType.clone(SmallVector<int64_t>{srcExtent, 1});
  } else {
    vecType = srcType.clone(SmallVector<int64_t>{1, srcExtent});
  }
  Value vec = rewriter.create<pypto::ReshapeOp>(loc, vecType, op.getSource(), ValueRange(vecExtents));
  std::string zero = isa<FloatType>(resultType.getElementType()) ? "0.0" : "0";
  Value carrier =
    rewriter.create<pypto::TileFullOp>(loc, convertedResult, rewriter.getStringAttr(zero), ValueRange(*extents))
      .getResult();
  Value result = rowLift ? rewriter.create<pypto::RowExpandAddOp>(loc, convertedResult, carrier, vec).getResult()
                         : rewriter.create<pypto::ColExpandAddOp>(loc, convertedResult, carrier, vec).getResult();
  rewriter.replaceOp(op, result);
  return success();
}

/// Rewrites a scalar broadcast:
/// - constant scalars materialize directly as `pl.tile.full(shape, dtype, v)`;
/// - runtime scalars are broadcast by adding them onto a zero carrier tile:
///   `pl.add(pl.tile.full(shape, dtype, 0), scalar)`. `pl.expands` has no
///   codegen on current device backends, while the scalar form of `pl.add`
///   broadcasts for free.
/// Dynamic `!npuvector` results take their `maxSizes` upper bounds as the
/// tile shape, matching the extent a dynamic transfer_read allocates.
/// A rank-1 tile source lifted onto a rank-2 result falls through to
/// `materializeLiftedBroadcast` — the tile twin of the runtime-scalar
/// zero-carrier trick; other tile-to-tile broadcasts have no PyPTO
/// counterpart.
template <typename BroadcastOp>
class BroadcastPattern : public OpConversionPattern<BroadcastOp> {
 public:
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BroadcastOp op, typename BroadcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "broadcast result must be shaped");
    }
    if constexpr (!std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
      if (!resultType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(op, "only statically-shaped broadcasts are supported");
      }
    }
    auto dtype = pyptoDtypeName(resultType.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
    }
    Type sourceType = op.getSource().getType();
    if (!isa<FloatType, IntegerType, IndexType>(sourceType)) {
      // A rank-1 tile source lifted onto a rank-2 result materializes as
      // a pure promotion; other tile-to-tile broadcasts have no PyPTO
      // counterpart.
      if constexpr (std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
        if (succeeded(materializeLiftedBroadcast(op, resultType, rewriter))) {
          return success();
        }
      }
      return rewriter.notifyMatchFailure(op, "only scalar broadcasts map onto pl.tile.full/pl.expands");
    }

    Location loc = op.getLoc();
    Type convertedResult = op->getResult(0).getType();

    // The tile's allocated extent: static dims stay in the result type;
    // dynamic dims take their `maxSizes` upper bound (constants fold into
    // index constants, runtime bounds stay operand values).
    FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
    if (failed(extents)) {
      return failure();
    }

    std::optional<std::string> constText;
    if (auto constOp = op.getSource().template getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        constText = floatToPythonText(floatAttr.getValue());
      } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        if (!intAttr.getType().isInteger(1)) {
          constText = intToPythonText(intAttr.getValue());
        }
      }
    }

    Value result;
    if (constText) {
      result =
        rewriter
          .create<pypto::TileFullOp>(loc, convertedResult, rewriter.getStringAttr(*constText), ValueRange(*extents))
          .getResult();
    } else {
      std::string zero = isa<FloatType>(sourceType) ? "0.0" : "0";
      Value carrier =
        rewriter.create<pypto::TileFullOp>(loc, convertedResult, rewriter.getStringAttr(zero), ValueRange(*extents))
          .getResult();
      result = rewriter.create<pypto::AddOp>(loc, convertedResult, carrier, op.getSource()).getResult();
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Fuses a rank-1 tile broadcast lifted to a tile's rank-2 shape into
/// every consumer: `arith.<op> %tile, %bcast` — where `%bcast` is an
/// `npuvector.broadcast {dimension = array<i64: 0>}` — becomes
/// `pypto.row_expand_<op>(%tile, reshape(%bcast.src))`, and
/// `{dimension = array<i64: 1>}` becomes the col_expand counterpart. The
/// runtime's expand families broadcast the reshaped vector across the
/// tile natively (valid semantics included), so the broadcast never
/// materializes. The reshape re-views the rank-1 source as `[rows, 1]`
/// (row lift) or `[1, cols]` (column lift); its extent mirrors the
/// source's allocation bound, which the broadcast's max sizes carry.
///
/// Every consumer must fuse — each is rewritten in place, so one exotic
/// consumer declines the whole fusion. Commutative consumers (add, mul)
/// accept the broadcast on either side; sub/div keep the rhs-only
/// contract because `expand_sub/div(tile, vec)` spells `tile - vec`.
/// Consumers may already sit in their converted pypto form; both forms
/// take the same tile-first expansion.
///
/// Like the scalar-broadcast fusion, the pattern is keyed on the
/// broadcast: the driver visits it before its consumers, and an
/// unlegalized broadcast aborts the whole conversion.
class TileBroadcastFusionPattern : public OpConversionPattern<npuvector::BroadcastOp> {
 public:
  using OpConversionPattern<npuvector::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, npuvector::BroadcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int64_t> dimension = op.getDimension();
    bool rowLift = dimension.size() == 1 && dimension[0] == 0;
    bool colLift = dimension.size() == 1 && dimension[0] == 1;
    if (!rowLift && !colLift) {
      return rewriter.notifyMatchFailure(op, "tile-broadcast fusion needs dimension = [0] or [1]");
    }
    auto srcType = dyn_cast<npuvector::NPUVectorType>(op.getSource().getType());
    if (!srcType || srcType.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "tile-broadcast fusion needs a rank-1 tile source");
    }
    // Every consumer must fuse; collect the tile operand each expands
    // against before rewriting anything.
    SmallVector<std::pair<Operation *, Value>> consumers;
    for (Operation *user : op->getUsers()) {
      Value tile = fusibleTileOperand(user, op.getResult());
      if (!tile) {
        return rewriter.notifyMatchFailure(op, "every consumer must fuse onto the expand family");
      }
      consumers.emplace_back(user, tile);
    }
    if (consumers.empty()) {
      return rewriter.notifyMatchFailure(op, "tile-broadcast fusion needs a consumer");
    }
    // The lifted axis is the identity mapping, so only a provable static
    // mismatch is a real error (dynamic `?` extents re-check at runtime).
    if (failed(checkLiftedExtents(rewriter, consumers, rowLift, srcType.getDimSize(0)))) {
      return failure();
    }
    // The reshaped vector: [rows, 1] for a row lift, [1, cols] for a
    // column lift. A dynamic extent takes the broadcast's max size.
    FailureOr<std::pair<Type, SmallVector<Value>>> lifted = buildLiftedVector(op, srcType, rowLift, rewriter);
    if (failed(lifted)) {
      return rewriter.notifyMatchFailure(op, "dynamic broadcast must carry per-dimension max sizes");
    }
    Value vec = rewriter.create<pypto::ReshapeOp>(op.getLoc(), lifted->first, op.getSource(),
                                                  ValueRange(lifted->second));
    for (auto [user, tile] : consumers) {
      // The expand lands at the consumer, not at the broadcast: the
      // tile operand may be defined between the two, and an expand
      // hoisted to the broadcast would then use it before its
      // definition.
      rewriter.setInsertionPoint(user);
      rewriter.replaceOp(user, makeExpand(user, rowLift, tile, vec, rewriter)->getResults());
    }
    rewriter.eraseOp(op);
    return success();
  }

 private:
  /// Checks every consumer's tile operand against the lifted source
  /// extent. The lifted axis is the identity mapping, so only a provable
  /// static mismatch is a real error; dynamic `?` extents re-check at
  /// runtime.
  static LogicalResult checkLiftedExtents(ConversionPatternRewriter &rewriter,
                                          ArrayRef<std::pair<Operation *, Value>> consumers, bool rowLift,
                                          int64_t srcExtent) {
    for (auto [user, tile] : consumers) {
      auto tileType = dyn_cast<ShapedType>(tile.getType());
      if (!tileType || tileType.getRank() != 2) {
        return rewriter.notifyMatchFailure(user, "expand fusion needs a rank-2 tile operand");
      }
      int64_t tileExtent = tileType.getDimSize(rowLift ? 0 : 1);
      if (!ShapedType::isDynamic(tileExtent) && !ShapedType::isDynamic(srcExtent) && tileExtent != srcExtent) {
        return rewriter.notifyMatchFailure(user, "lifted extents do not match");
      }
    }
    return success();
  }

  /// Builds the reshaped rank-2 type and runtime extents for the lifted
  /// vector: `[rows, 1]` for a row lift, `[1, cols]` for a column lift.
  /// A dynamic extent takes the broadcast's aligned max size; a static
  /// one mirrors the source extent and carries no runtime bounds.
  static FailureOr<std::pair<Type, SmallVector<Value>>> buildLiftedVector(npuvector::BroadcastOp op,
                                                                         npuvector::NPUVectorType srcType,
                                                                         bool rowLift,
                                                                         ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    int64_t srcExtent = srcType.getDimSize(0);
    if (ShapedType::isDynamic(srcExtent)) {
      ValueRange maxSizes = op.getMaxSizes();
      if (maxSizes.size() != 2) {
        return failure();
      }
      Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      // Pad a constant max size to whole 32-byte rows, matching the
      // aligned extent a dynamic transfer_read allocates, so the
      // reshape stays shape-compatible with the source tile (42 -> 48
      // for FP32). Mirrors `buildBroadcastExtents`.
      auto alignedMax = [&](Value max) -> Value {
        if (auto bound = constIndexValue(max)) {
          unsigned elemBytes = srcType.getElementType().getIntOrFloatBitWidth() / 8;
          return rewriter.create<arith::ConstantIndexOp>(loc, alignExtentTo32B(*bound, elemBytes));
        }
        return max;
      };
      if (rowLift) {
        Type vecType = srcType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
        return std::make_pair(vecType, SmallVector<Value>{alignedMax(maxSizes[0]), one});
      }
      Type vecType = srcType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
      return std::make_pair(vecType, SmallVector<Value>{one, alignedMax(maxSizes[1])});
    }
    if (rowLift) {
      Type vecType = srcType.clone(SmallVector<int64_t>{srcExtent, 1});
      return std::make_pair(vecType, SmallVector<Value>{});
    }
    Type vecType = srcType.clone(SmallVector<int64_t>{1, srcExtent});
    return std::make_pair(vecType, SmallVector<Value>{});
  }

  /// The tile operand `user` expands the broadcast against, or none when
  /// `user` cannot fuse. Commutative consumers accept the broadcast on
  /// either side; sub/div keep the rhs-only contract.
  static Value fusibleTileOperand(Operation *user, Value bcast) {
    bool commutative = isa<arith::AddFOp, arith::MulFOp, pypto::AddOp, pypto::MulOp>(user);
    if (!commutative && !isa<arith::SubFOp, arith::DivFOp, pypto::SubOp, pypto::DivOp>(user)) {
      return Value();
    }
    Value lhs = user->getOperand(0);
    Value rhs = user->getOperand(1);
    if (lhs == bcast && rhs == bcast) {
      return Value();
    }
    if (rhs == bcast) {
      return lhs;
    }
    if (commutative && lhs == bcast) {
      return rhs;
    }
    return Value();
  }

  /// Builds the expand op matching `user`'s arithmetic kind, in the row
  /// or column family the lift selects.
  static Operation *makeExpand(Operation *user, bool rowLift, Value tile, Value vec,
                               ConversionPatternRewriter &rewriter) {
    Location loc = user->getLoc();
    Type result = user->getResult(0).getType();
    if (isa<arith::AddFOp, pypto::AddOp>(user)) {
      return rowLift ? rewriter.create<pypto::RowExpandAddOp>(loc, result, tile, vec).getOperation()
                     : rewriter.create<pypto::ColExpandAddOp>(loc, result, tile, vec).getOperation();
    }
    if (isa<arith::SubFOp, pypto::SubOp>(user)) {
      return rowLift ? rewriter.create<pypto::RowExpandSubOp>(loc, result, tile, vec).getOperation()
                     : rewriter.create<pypto::ColExpandSubOp>(loc, result, tile, vec).getOperation();
    }
    if (isa<arith::MulFOp, pypto::MulOp>(user)) {
      return rowLift ? rewriter.create<pypto::RowExpandMulOp>(loc, result, tile, vec).getOperation()
                     : rewriter.create<pypto::ColExpandMulOp>(loc, result, tile, vec).getOperation();
    }
    return rowLift ? rewriter.create<pypto::RowExpandDivOp>(loc, result, tile, vec).getOperation()
                   : rewriter.create<pypto::ColExpandDivOp>(loc, result, tile, vec).getOperation();
  }
};

/// Rewrites a rank-1 comparison broadcast into a full-geometry
/// comparison: `broadcast{dimension = [0]}(cmp %vec, %scalar)` lifts the
/// comparison's tile operands to the broadcast's shape and re-creates
/// the `pypto.cmp` there. The lift is a pure promotion — a zero tile
/// plus the runtime's row/col expand add spells `0 + broadcast(%vec)`,
/// which cannot change the compared values — because `pl.cmp` demands
/// same-shaped tiles (it never broadcasts) while `pl.tile.select`
/// demands a condition at its own geometry, so the packed predicate
/// mask has to be rebuilt at the consumer's shape instead of broadcast.
///
/// The source comparison may already sit in its converted `pypto.cmp`
/// form — the scalar-broadcast fusion runs ahead and rewrites
/// `cmpf tile, broadcast(scalar)` onto the tile-scalar form — or still
/// be an npuvector cmpi/cmpf; both spellings are recognized. A scalar
/// rhs — directly or behind a scalar broadcast — passes through (it
/// re-rides the tile-scalar `pl.tile.cmps`, and never references the
/// broadcast result the scalar-broadcast fusion may erase); a tile rhs
/// lifts alongside the lhs. The source comparison stays in place for
/// its remaining users, if any.
class BroadcastCmpMaskPattern : public OpConversionPattern<npuvector::BroadcastOp> {
 public:
  using OpConversionPattern<npuvector::BroadcastOp>::OpConversionPattern;

  /// The comparison predicate feeding the broadcast, in either spelling
  /// (`pl.cmp`, `npuvector.cmpi/cmpf`), or none when `pl.cmp` cannot
  /// spell it (NaN-sensitive floats). The comparison's operands ride
  /// along as the source's first two operands.
  static std::optional<pypto::CmpPredicate> cmpPredicateFor(Operation *source) {
    if (auto cmp = dyn_cast<pypto::CmpOp>(source)) {
      return cmp.getPredicate();
    }
    if (auto cmp = dyn_cast<npuvector::CmpIOp>(source)) {
      if (auto code = cmpTypeCodeFor(cmp.getPredicate())) {
        return static_cast<pypto::CmpPredicate>(*code);
      }
    }
    if (auto cmp = dyn_cast<npuvector::CmpFOp>(source)) {
      if (auto code = cmpTypeCodeFor(cmp.getPredicate())) {
        return static_cast<pypto::CmpPredicate>(*code);
      }
    }
    return std::nullopt;
  }

  /// Reshapes `tile` to `[extent, 1]` (row lift) or `[1, extent]`
  /// (column lift) and expands it onto the zero carrier's geometry.
  static Value liftOntoCarrier(Location loc, npuvector::NPUVectorType valueTileType, Value carrier, bool rowLift,
                               Value tile, ConversionPatternRewriter &rewriter) {
    auto tileType = cast<npuvector::NPUVectorType>(tile.getType());
    Type vecType = rowLift ? tileType.clone(SmallVector<int64_t>{tileType.getDimSize(0), 1})
                           : tileType.clone(SmallVector<int64_t>{1, tileType.getDimSize(0)});
    Value vec = rewriter.create<pypto::ReshapeOp>(loc, vecType, tile, ValueRange());
    if (rowLift) {
      return rewriter.create<pypto::RowExpandAddOp>(loc, valueTileType, carrier, vec).getResult();
    }
    return rewriter.create<pypto::ColExpandAddOp>(loc, valueTileType, carrier, vec).getResult();
  }

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, npuvector::BroadcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int64_t> dimension = op.getDimension();
    bool rowLift = dimension.size() == 1 && dimension[0] == 0;
    bool colLift = dimension.size() == 1 && dimension[0] == 1;
    if (!rowLift && !colLift) {
      return rewriter.notifyMatchFailure(op, "cmp-mask lift needs dimension = [0] or [1]");
    }
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "cmp-mask lift needs a statically-shaped rank-2 result");
    }
    // The comparison feeding the broadcast, in either spelling; a
    // predicate `pl.cmp` cannot spell (NaN-sensitive floats) declines.
    Operation *source = op.getSource().getDefiningOp();
    std::optional<pypto::CmpPredicate> predicate = cmpPredicateFor(source);
    if (!predicate) {
      return rewriter.notifyMatchFailure(op, "broadcast source is not a comparison pl.cmp can spell");
    }
    Value cmpLhs = source->getOperand(0);
    Value cmpRhs = source->getOperand(1);
    auto lhsType = dyn_cast<npuvector::NPUVectorType>(cmpLhs.getType());
    if (!lhsType || lhsType.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "cmp-mask lift needs a rank-1 tile lhs");
    }
    // The lifted axis is the identity mapping, so a static mismatch is
    // a real error.
    if (resultType.getDimSize(rowLift ? 0 : 1) != lhsType.getDimSize(0)) {
      return rewriter.notifyMatchFailure(op, "lifted extents do not match");
    }
    Location loc = op.getLoc();
    // The value geometry: the broadcast's shape over the comparison's
    // element type. The lift below promotes a rank-1 tile onto it.
    auto valueTileType = cast<npuvector::NPUVectorType>(lhsType.clone(resultType.getShape()));
    std::string zero = isa<FloatType>(lhsType.getElementType()) ? "0.0" : "0";
    Value carrier =
      rewriter.create<pypto::TileFullOp>(loc, valueTileType, rewriter.getStringAttr(zero), ValueRange()).getResult();
    Value liftedLhs = liftOntoCarrier(loc, valueTileType, carrier, rowLift, cmpLhs, rewriter);
    // A scalar rhs re-rides the tile-scalar comparison; a tile rhs lifts.
    // A rhs that merely broadcasts a scalar passes through to that
    // scalar: lifting the broadcast result would hang a reshape off an
    // op the scalar-broadcast fusion erases, whichever pattern runs
    // first.
    Value finalRhs = cmpRhs;
    if (auto rhsBroadcast = cmpRhs.getDefiningOp<npuvector::BroadcastOp>()) {
      if (!isa<VectorType, npuvector::NPUVectorType>(rhsBroadcast.getSource().getType())) {
        finalRhs = rhsBroadcast.getSource();
      }
    }
    if (isa<VectorType, npuvector::NPUVectorType>(finalRhs.getType())) {
      finalRhs = liftOntoCarrier(loc, valueTileType, carrier, rowLift, finalRhs, rewriter);
    }
    Value mask =
      rewriter.create<pypto::CmpOp>(loc, op->getResult(0).getType(), *predicate, liftedLhs, finalRhs);
    replaceWithPackedMask(op, mask, rewriter);
    return success();
  }
};

/// Fuses a scalar broadcast into every fusable consumer: `cmp pred,
/// %tile, broadcast(%scalar)` becomes `pypto.cmp pred, %tile, %scalar`
/// (lowered onto the tile-scalar `pl.tile.cmps`), and
/// `select(%cond, broadcast(%scalar), %tile)` becomes `pypto.select
/// (%cond, %scalar, %tile)` (the composite `pl.tile.select` accepts a
/// literal scalar branch and synthesizes its own scratch). A
/// materialized broadcast tile cannot follow the loads' shape convention
/// (leading 1s to the source rank plus 32B-aligned columns — the alignment
/// is dtype-relative, so an f32 carrier drifts from a bf16-allocated load
/// cast into f32), so the scalar forms sidestep tile-shape matching
/// entirely.
///
/// Like the row-expand fusion, the pattern is keyed on the broadcast: the
/// driver visits it before its consumers, and an unlegalized broadcast
/// aborts the whole conversion. Every use must be fusable; one exotic
/// consumer (a non-binary arith operand, a comparison lhs, the second
/// branch of a scalar-branch select) declines the whole fusion and falls
/// back to the tile_full materialization.
class ScalarBroadcastFusionPattern : public OpConversionPattern<npuvector::BroadcastOp> {
 public:
  using OpConversionPattern<npuvector::BroadcastOp>::OpConversionPattern;

  /// Binary arith ops that map onto a pypto tile binary (mirroring the
  /// OpToPyptoOpPattern registrations).
  static bool isFusableArithBinary(Operation *op) {
    return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::AddIOp, arith::SubIOp,
               arith::MulIOp, arith::DivSIOp, arith::DivUIOp>(op);
  }

  /// Commutative binaries may take the broadcast on either side; the
  /// operands swap into the runtime's tile-first form.
  static bool isCommutativeArith(Operation *op) {
    return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(op);
  }

  /// Match gate for one broadcast user: comparisons must take the
  /// broadcast as their tile rhs (with a predicate `pl.cmp` can spell);
  /// selects must take it on exactly one branch with tile operands
  /// beside it; binary arith ops must take it as the rhs of a tile (or
  /// the commutative lhs).
  static bool isFusibleUser(Operation *user, Value bcast, bool isTileType(Type)) {
    if (auto cmp = dyn_cast<npuvector::CmpIOp>(user)) {
      return cmp.getRhs() == bcast && isTileType(cmp.getLhs().getType()) && cmpTypeCodeFor(cmp.getPredicate());
    }
    if (auto cmp = dyn_cast<npuvector::CmpFOp>(user)) {
      return cmp.getRhs() == bcast && isTileType(cmp.getLhs().getType()) && cmpTypeCodeFor(cmp.getPredicate());
    }
    if (auto select = dyn_cast<npuvector::SelectOp>(user)) {
      bool trueIsBcast = select.getTrueValue() == bcast;
      bool falseIsBcast = select.getFalseValue() == bcast;
      if (trueIsBcast == falseIsBcast) {
        return false;
      }
      Value other = trueIsBcast ? select.getFalseValue() : select.getTrueValue();
      return isTileType(other.getType()) && isTileType(select.getCondition().getType());
    }
    if (isFusableArithBinary(user)) {
      Value lhs = user->getOperand(0);
      Value rhs = user->getOperand(1);
      if (rhs == bcast) {
        return isTileType(lhs.getType());
      }
      if (lhs == bcast && isCommutativeArith(user)) {
        return isTileType(rhs.getType());
      }
    }
    return false;
  }

  /// The composite `pl.tile.select` materializes a scalar branch with
  /// `tile.full`, which only accepts literals — so a broadcast feeding a
  /// select is only fusable when its scalar source is one. Both constant
  /// spellings are accepted: the raw `arith.constant`, and the
  /// `pypto.constant` it rewrites into (pattern order is not fixed).
  static bool isLiteralScalar(Value scalar) {
    if (scalar.getDefiningOp<pypto::ConstantOp>()) {
      return static_cast<bool>(pyptoDtypeName(scalar.getType()));
    }
    auto arithConst = scalar.getDefiningOp<arith::ConstantOp>();
    if (!arithConst || !pyptoDtypeName(scalar.getType())) {
      return false;
    }
    if (dyn_cast<FloatAttr>(arithConst.getValue())) {
      return true;  // floatToPythonText renders any float
    }
    if (auto intAttr = dyn_cast<IntegerAttr>(arithConst.getValue())) {
      return !intAttr.getType().isInteger(1) && intToPythonText(intAttr.getValue());
    }
    return false;
  }

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, npuvector::BroadcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // NPUVectorType only implements ShapedTypeInterface — it is not a
    // ShapedType subclass — so tile-ness must be probed through the
    // concrete pair like the conversion target's legality lambda does.
    auto isTileType = [](Type type) { return isa<VectorType, npuvector::NPUVectorType>(type); };
    if (isTileType(op.getSource().getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "scalar fusion needs a scalar "
                                         "source");
    }
    // Match phase: every consumer must compare the broadcast as its rhs
    // or select it on exactly one branch; a select additionally demands
    // a literal scalar source (see isLiteralScalar).
    SmallVector<Operation *> users(op->getUsers());
    if (llvm::any_of(users, [&](Operation *user) { return !isFusibleUser(user, op.getResult(), isTileType); })) {
      return rewriter.notifyMatchFailure(op,
                                         "scalar fusion only feeds tile-rhs comparisons, "
                                         "one-branch selects, and binary arith ops");
    }
    if (llvm::any_of(users, [](Operation *user) { return isa<npuvector::SelectOp>(user); }) &&
        !isLiteralScalar(op.getSource())) {
      return rewriter.notifyMatchFailure(op, "select fusion needs a literal scalar with a PyPTO dtype");
    }
    // Rewrite phase: swap each consumer onto the scalar form in place,
    // then drop the broadcast. New ops are built at the consumer's own
    // position — the driver's insertion point stays at the broadcast, so
    // leaving it there would pile every replacement before the broadcast
    // in creation order and break dominance when a select is created
    // ahead of the cmp defining its condition.
    Value scalar = op.getSource();
    for (Operation *user : users) {
      rewriter.setInsertionPoint(user);
      if (auto cmp = dyn_cast<npuvector::CmpIOp>(user)) {
        auto code = cmpTypeCodeFor(cmp.getPredicate());
        Value mask = rewriter.create<pypto::CmpOp>(cmp.getLoc(), cmp->getResult(0).getType(),
                                                   static_cast<pypto::CmpPredicate>(*code), cmp.getLhs(), scalar);
        replaceWithPackedMask(cmp, mask, rewriter);
      } else if (auto cmp = dyn_cast<npuvector::CmpFOp>(user)) {
        auto code = cmpTypeCodeFor(cmp.getPredicate());
        Value mask = rewriter.create<pypto::CmpOp>(cmp.getLoc(), cmp->getResult(0).getType(),
                                                   static_cast<pypto::CmpPredicate>(*code), cmp.getLhs(), scalar);
        replaceWithPackedMask(cmp, mask, rewriter);
      } else if (isFusableArithBinary(user)) {
        // `tile OP scalar`: the broadcast sits on the rhs, or (commutative
        // ops) on the lhs with the operands swapped into the runtime's
        // tile-first form.
        Value tile = user->getOperand(1) == op.getResult() ? user->getOperand(0) : user->getOperand(1);
        if (isa<arith::AddFOp, arith::AddIOp>(user)) {
          rewriter.replaceOpWithNewOp<pypto::AddOp>(user, user->getResultTypes(), tile, scalar);
        } else if (isa<arith::SubFOp, arith::SubIOp>(user)) {
          rewriter.replaceOpWithNewOp<pypto::SubOp>(user, user->getResultTypes(), tile, scalar);
        } else if (isa<arith::MulFOp, arith::MulIOp>(user)) {
          rewriter.replaceOpWithNewOp<pypto::MulOp>(user, user->getResultTypes(), tile, scalar);
        } else {
          rewriter.replaceOpWithNewOp<pypto::DivOp>(user, user->getResultTypes(), tile, scalar);
        }
      } else {
        auto select = cast<npuvector::SelectOp>(user);
        bool trueIsBcast = select.getTrueValue() == op.getResult();
        rewriter.replaceOpWithNewOp<pypto::SelectOp>(select, select->getResult(0).getType(), select.getCondition(),
                                                     trueIsBcast ? scalar : select.getTrueValue(),
                                                     trueIsBcast ? select.getFalseValue() : scalar);
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrites a transpose into `pypto.transpose`: PyPTO swaps exactly
/// two axes, so the permutation must be a single transposition (or the
/// identity, which folds away).
template <typename TransposeOp>
class TransposePattern : public OpConversionPattern<TransposeOp> {
 public:
  using OpConversionPattern<TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(TransposeOp op, typename TransposeOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int64_t> permutation = op.getPermutation();
    std::optional<std::pair<int64_t, int64_t>> swap;
    for (auto [axis, target] : llvm::enumerate(permutation)) {
      if (target == static_cast<int64_t>(axis)) {
        continue;
      }
      if (swap) {
        // The far end of the same transposition is expected; anything else
        // is a second, independent displacement.
        if (static_cast<int64_t>(axis) == swap->second && target == swap->first) {
          continue;
        }
        return rewriter.notifyMatchFailure(op, "permutation is not a single axis swap");
      }
      if (target < 0 || target >= static_cast<int64_t>(permutation.size()) ||
          permutation[target] != static_cast<int64_t>(axis)) {
        return rewriter.notifyMatchFailure(op, "permutation is not a single axis swap");
      }
      swap = std::make_pair(static_cast<int64_t>(axis), target);
    }
    if (!swap) {
      // Identity permutation: the value passes through unchanged.
      rewriter.replaceOp(op, op.getVector());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(op, op->getResult(0).getType(), op.getVector(),
                                                    static_cast<uint64_t>(swap->first),
                                                    static_cast<uint64_t>(swap->second));
    return success();
  }
};

/// Rewrites a select into `pypto.select`: the runtime's composite
/// `pl.tile.select` synthesizes the architecture-defined scratch tile, so
/// no scratch is materialized here and dynamic result shapes are legal.
/// Constant scalar branches are materialized as `pypto.tile_full` (which
/// needs static shapes, since the extents would otherwise be unknown);
/// tile branches keep the source shaped type. Scalar selects (e.g. the
/// branchless floordiv chain left by the built-in affine lowering) keep
/// the arith op; convert-pypto-to-emitpy lowers it onto the `select`
/// pseudo-callee, which the translator prints as a Python conditional
/// expression.
template <typename SelectOp>
class SelectPattern : public OpConversionPattern<SelectOp> {
 public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SelectOp op, typename SelectOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Scalar selects keep the arith op (see the class comment): stage two
    // maps them onto the `select` pseudo-callee. A scalar npuvector
    // select cannot stay in the source dialect: rewrite onto arith.
    if (!isa<ShapedType>(op->getResult(0).getType())) {
      if constexpr (std::is_same_v<SelectOp, npuvector::SelectOp>) {
        rewriter.replaceOpWithNewOp<arith::SelectOp>(op, op.getCondition(), op.getTrueValue(), op.getFalseValue());
        return success();
      }
      return rewriter.notifyMatchFailure(op, "scalar select stays arith");
    }
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "select result must be a tile");
    }
    if (!pyptoDtypeName(resultType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
    }
    if (!isa<ShapedType>(op.getCondition().getType())) {
      return rewriter.notifyMatchFailure(op, "condition must be a tile");
    }

    Location loc = op.getLoc();
    Type convertedResult = op->getResult(0).getType();
    bool staticShape = resultType.hasStaticShape();

    // Constant scalar branches become full tiles; that materialization
    // needs a static shape, since a dynamic tile_full would have no
    // extents to bind.
    auto materializeBranch = [&](Value branch) -> Value {
      if (isa<ShapedType>(branch.getType())) {
        return branch;
      }
      if (!staticShape) {
        return Value();
      }
      auto constOp = branch.getDefiningOp<arith::ConstantOp>();
      if (!constOp) {
        return Value();
      }
      std::optional<std::string> text;
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        text = floatToPythonText(floatAttr.getValue());
      } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        if (!intAttr.getType().isInteger(1)) {
          text = intToPythonText(intAttr.getValue());
        }
      }
      if (!text) {
        return Value();
      }
      return rewriter.create<pypto::TileFullOp>(loc, convertedResult, rewriter.getStringAttr(*text), ValueRange{})
        .getResult();
    };

    Value trueBranch = materializeBranch(op.getTrueValue());
    Value falseBranch = materializeBranch(op.getFalseValue());
    if (!trueBranch || !falseBranch) {
      return rewriter.notifyMatchFailure(op,
                                         "select branches must be tiles, or constant scalars on a "
                                         "statically shaped select");
    }

    // The select keeps the source result type so the conversion never
    // materializes casts; convert-pypto-to-emitpy lowers it into the
    // positional `pl.tile.select` composite call.
    rewriter.replaceOpWithNewOp<pypto::SelectOp>(op, convertedResult, op.getCondition(), trueBranch, falseBranch);
    return success();
  }
};

/// Rewrites `memref.collapse_shape` into `pypto.reshape`: the flattened
/// tensor view PyPTO loads and stores through. Only statically shaped
/// views are supported; the shape travels in the result type and
/// convert-pypto-to-emitpy re-derives the `pl.reshape(src, [shape])`
/// argument from it.
class CollapseShapePattern : public OpConversionPattern<memref::CollapseShapeOp> {
 public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only statically shaped collapse views are supported");
    }
    // The replacement keeps the original result type: the type converter is
    // the identity, so a type-changing replacement would demand a
    // materialization at already-converted users (e.g. a live `pl.load`) and
    // abort the full conversion. The raw source operand is used on purpose:
    // signature rewriting happens by hand after the conversion, so the
    // memref stays as-is.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(), op.getSrc());
    return success();
  }
};

/// Rewrites `memref.expand_shape` into `pypto.reshape`, the mirror of the
/// collapse handling. Only statically shaped views are supported.
class ExpandShapePattern : public OpConversionPattern<memref::ExpandShapeOp> {
 public:
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only statically shaped expand views are supported");
    }
    // Same materialization constraints as the collapse handling: the
    // replacement keeps the original result type.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(), op.getSrc());
    return success();
  }
};

/// Rewrites `memref.transpose` into `pypto.transpose`: PyPTO swaps
/// exactly two axes, so the permutation must be a single transposition (or
/// the identity, which folds away). The memref permutation is an AffineMap
/// whose result dim i comes from source dim `getDimPosition(i)`, the same
/// convention as the vector transpose.
class MemrefTransposePattern : public OpConversionPattern<memref::TransposeOp> {
 public:
  using OpConversionPattern<memref::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    AffineMap permutation = op.getPermutation();
    SmallVector<int64_t> mapping(permutation.getNumResults());
    for (auto [axis, dim] : llvm::enumerate(mapping)) {
      dim = permutation.getDimPosition(axis);
    }
    std::optional<std::pair<int64_t, int64_t>> swap;
    for (auto [axis, target] : llvm::enumerate(mapping)) {
      if (target == static_cast<int64_t>(axis)) {
        continue;
      }
      if (swap) {
        // The far end of the same transposition is expected; anything else
        // is a second, independent displacement.
        if (static_cast<int64_t>(axis) == swap->second && target == swap->first) {
          continue;
        }
        return rewriter.notifyMatchFailure(op, "permutation is not a single axis swap");
      }
      if (target < 0 || target >= static_cast<int64_t>(mapping.size()) ||
          mapping[target] != static_cast<int64_t>(axis)) {
        return rewriter.notifyMatchFailure(op, "permutation is not a single axis swap");
      }
      swap = std::make_pair(static_cast<int64_t>(axis), target);
    }
    if (!swap) {
      // Identity permutation: the value passes through unchanged.
      rewriter.replaceOp(op, op.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(op, op.getResult().getType(), op.getIn(),
                                                    static_cast<uint64_t>(swap->first),
                                                    static_cast<uint64_t>(swap->second));
    return success();
  }
};

/// Rewrites `memref.reshape` into `pypto.reshape`, matching the
/// collapse/expand handling. Only statically shaped results are supported;
/// the shape operand itself becomes dead and is swept by
/// convert-pypto-to-emitpy.
class ReshapePattern : public OpConversionPattern<memref::ReshapeOp> {
 public:
  using OpConversionPattern<memref::ReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only statically shaped reshapings are supported");
    }
    // Same materialization constraints as the collapse handling: the
    // replacement keeps the original result type.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(), op.getSource());
    return success();
  }
};

/// Rewrites `memref.subview` into `pypto.slice`: the window is the
/// subview's sizes (full-rank over the source, with unit dims standing in
/// for the axes a rank-reducing view drops) and the offsets carry any
/// dynamic entries. Non-unit strides have no PyPTO lowering.
class SubviewPattern : public OpConversionPattern<memref::SubViewOp> {
 public:
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only statically shaped subviews are supported");
    }
    auto srcType = dyn_cast<ShapedType>(op.getSource().getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(op, "unranked sources are unsupported");
    }
    for (OpFoldResult stride : op.getMixedStrides()) {
      auto cst = staticIntValue(stride);
      if (!cst || *cst != 1) {
        return rewriter.notifyMatchFailure(op, "only unit-stride subviews are supported");
      }
    }
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<int64_t> window(resultType.getShape().begin(), resultType.getShape().end());
    SmallVector<unsigned> droppedDims;
    if (resultType.getRank() < srcType.getRank()) {
      // A rank-reducing view: recover the full-rank window from the sizes,
      // where the dropped axes appear as unit dims, and record them for the
      // drop_dims argument of pl.slice.
      SmallVector<int64_t> sizes;
      for (OpFoldResult size : op.getMixedSizes()) {
        auto cst = staticIntValue(size);
        if (!cst) {
          return rewriter.notifyMatchFailure(op, "rank-reducing subviews need static sizes");
        }
        sizes.push_back(*cst);
      }
      auto dropped = computeRankReductionMask(sizes, resultType.getShape());
      if (!dropped) {
        return rewriter.notifyMatchFailure(op, "cannot map the rank-reduced window onto pl.slice");
      }
      window = std::move(sizes);
      droppedDims.assign(dropped->begin(), dropped->end());
      llvm::sort(droppedDims);
    }
    replaceWithSliceOp(op, op.getSource(), op.getResult(), window, offsets, droppedDims, rewriter);
    return success();
  }
};

/// Builds the full-rank slice coordinates for a reinterpret cast with a
/// static flat offset.
static LogicalResult buildStaticReinterpretWindow(memref::ReinterpretCastOp op, MemRefType srcType,
                                                  MemRefType resultType, ArrayRef<int64_t> srcStrides,
                                                  ConversionPatternRewriter &rewriter,
                                                  SmallVectorImpl<OpFoldResult> &offsets,
                                                  SmallVectorImpl<int64_t> &window,
                                                  SmallVectorImpl<unsigned> &droppedDims) {
  int64_t rem = *staticIntValue(op.getMixedOffsets().front());
  unsigned srcRank = srcType.getRank();
  unsigned resultRank = resultType.getRank();
  for (unsigned axis = 0; axis < srcRank; ++axis) {
    int64_t coord = rem / srcStrides[axis];
    rem %= srcStrides[axis];
    offsets.push_back(rewriter.getIndexAttr(coord));
    if (axis < srcRank - resultRank) {
      window.push_back(1);
      droppedDims.push_back(axis);
    } else {
      window.push_back(resultType.getDimSize(axis - (srcRank - resultRank)));
    }
  }
  if (rem != 0) {
    return rewriter.notifyMatchFailure(op, "offset does not decompose into source coordinates");
  }
  return success();
}

/// Builds the full-rank slice coordinates for a reinterpret cast with a
/// dynamic flat offset.
static LogicalResult buildDynamicReinterpretWindow(memref::ReinterpretCastOp op, MemRefType srcType,
                                                   MemRefType resultType, ArrayRef<OpFoldResult> strides,
                                                   ConversionPatternRewriter &rewriter,
                                                   SmallVectorImpl<OpFoldResult> &offsets,
                                                   SmallVectorImpl<int64_t> &window,
                                                   SmallVectorImpl<unsigned> &droppedDims) {
  auto stride0 = staticIntValue(strides[0]);
  if (resultType.getRank() != 1 || !stride0 || *stride0 != 1) {
    return rewriter.notifyMatchFailure(op, "dynamic offsets only map onto the last source axis");
  }
  unsigned srcRank = srcType.getRank();
  OpFoldResult flatOffset = op.getMixedOffsets().front();
  for (unsigned axis = 0; axis < srcRank; ++axis) {
    if (axis + 1 == srcRank) {
      offsets.push_back(flatOffset);
      window.push_back(resultType.getDimSize(0));
    } else {
      offsets.push_back(rewriter.getIndexAttr(0));
      window.push_back(1);
      droppedDims.push_back(axis);
    }
  }
  return success();
}

/// Rewrites `memref.reinterpret_cast` into `pypto.slice` when the cast
/// re-windows a region of its source that follows the source's own
/// trailing strides: the flat element offset (counted from the source's
/// base, per the descriptor semantics of the op) decomposes into per-axis
/// coordinates. Dynamic offsets only map when the window runs along the
/// last source axis, where the flat offset is that axis's coordinate.
class ReinterpretCastPattern : public OpConversionPattern<memref::ReinterpretCastOp> {
 public:
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op, "only statically shaped reinterpretations are supported");
    }
    if (!srcType) {
      return rewriter.notifyMatchFailure(op, "unranked sources are unsupported");
    }
    unsigned srcRank = srcType.getRank();
    unsigned resultRank = resultType.getRank();
    if (resultRank == 0 || resultRank > srcRank) {
      return rewriter.notifyMatchFailure(op, "result rank must fit inside the source");
    }
    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = 0;
    if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)) || srcOffset != 0) {
      return rewriter.notifyMatchFailure(op, "only zero-offset strided sources are supported");
    }
    // The result axes must walk the source's trailing axes one element at a
    // time for pl.slice to express the window.
    auto strides = op.getMixedStrides();
    for (unsigned axis = 0; axis < resultRank; ++axis) {
      auto cst = staticIntValue(strides[axis]);
      if (!cst || *cst != srcStrides[srcRank - resultRank + axis]) {
        return rewriter.notifyMatchFailure(op, "strides must match the source's trailing layout");
      }
    }
    SmallVector<OpFoldResult> offsets;
    SmallVector<int64_t> window;
    SmallVector<unsigned> droppedDims;
    if (staticIntValue(op.getMixedOffsets().front())) {
      if (failed(buildStaticReinterpretWindow(op, srcType, resultType, srcStrides, rewriter, offsets, window,
                                              droppedDims))) {
        return failure();
      }
    } else if (failed(buildDynamicReinterpretWindow(op, srcType, resultType, strides, rewriter, offsets, window,
                                                    droppedDims))) {
      return failure();
    }
    replaceWithSliceOp(op, op.getSource(), op.getResult(), window, offsets, droppedDims, rewriter);
    return success();
  }
};

/// Rewrites an NPU vector extract_slice into the PyPTO slice view. Dynamic
/// slice sizes use the source tile's static allocation bound as the window;
/// the result type remains dynamic so tail extents continue through the
/// vectorized arithmetic and memory operations.
class ExtractSlicePattern : public OpConversionPattern<npuvector::ExtractSliceOp> {
 public:
  using OpConversionPattern<npuvector::ExtractSliceOp>::OpConversionPattern;

  /// Computes the per-dimension slice window of unit strides: the static
  /// size when the slice spells one, otherwise the source's whole bound.
  /// Fails on a non-unit stride, a dynamic source bound, or a size that
  /// escapes the bound.
  static LogicalResult sliceWindowFor(ConversionPatternRewriter &rewriter, npuvector::ExtractSliceOp op,
                                      npuvector::NPUVectorType sourceType, SmallVectorImpl<int64_t> &window) {
    window.reserve(sourceType.getRank());
    for (unsigned dim = 0; dim < sourceType.getRank(); ++dim) {
      if (auto stride = constIndexValue(op.getStrides()[dim]); !stride || *stride != 1) {
        return rewriter.notifyMatchFailure(op, "only unit strides are supported");
      }
      int64_t bound = sourceType.getDimSize(dim);
      if (ShapedType::isDynamic(bound)) {
        return rewriter.notifyMatchFailure(op, "dynamic source bounds are unsupported");
      }
      if (auto size = constIndexValue(op.getSizes()[dim]); size) {
        if (*size <= 0 || *size > bound) {
          return rewriter.notifyMatchFailure(op, "slice size exceeds source bound");
        }
        window.push_back(*size);
      } else {
        window.push_back(bound);
      }
    }
    return success();
  }

  LogicalResult matchAndRewrite(npuvector::ExtractSliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto sourceType = dyn_cast<npuvector::NPUVectorType>(op.getSource().getType());
    auto resultType = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!sourceType || !resultType) {
      return rewriter.notifyMatchFailure(op, "extract_slice requires NPU vector types");
    }
    if (op.getStrides().size() != sourceType.getRank() || op.getSizes().size() != sourceType.getRank() ||
        op.getOffsets().size() != sourceType.getRank()) {
      return rewriter.notifyMatchFailure(op, "slice ranges must match source rank");
    }
    SmallVector<int64_t> window;
    if (failed(sliceWindowFor(rewriter, op, sourceType, window))) {
      return failure();
    }
    SmallVector<unsigned> droppedDims;
    llvm::SmallBitVector kept(sourceType.getRank());
    for (int64_t dim : op.getKeepDims()) {
      if (dim < 0 || dim >= static_cast<int64_t>(sourceType.getRank())) {
        return rewriter.notifyMatchFailure(op, "keep_dims contains an out-of-range dimension");
      }
      kept.set(dim);
    }
    for (unsigned dim = 0; dim < sourceType.getRank(); ++dim) {
      if (!kept.test(dim)) {
        if (window[dim] != 1) {
          return rewriter.notifyMatchFailure(op, "dropped dimensions must have unit size");
        }
        droppedDims.push_back(dim);
      }
    }
    if (kept.count() != resultType.getRank()) {
      return rewriter.notifyMatchFailure(op, "keep_dims does not match result rank");
    }
    DenseI64ArrayAttr dropDimsAttr;
    if (!droppedDims.empty()) {
      dropDimsAttr = rewriter.getDenseI64ArrayAttr(llvm::to_vector_of<int64_t>(droppedDims));
    }
    rewriter.replaceOpWithNewOp<pypto::SliceOp>(op, op.getResult().getType(), op.getSource(), window,
                                                adaptor.getOffsets(), dropDimsAttr);
    return success();
  }
};

/// Rewrites an NPU vector insert_slice when its result is immediately stored
/// back to the destination. The transfer_write retains the dynamic slice
/// extent, so forwarding the inserted source preserves tail semantics.
class InsertSlicePattern : public OpConversionPattern<npuvector::InsertSliceOp> {
 public:
  using OpConversionPattern<npuvector::InsertSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::InsertSliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op->getUsers().empty() || llvm::any_of(op->getUsers(), [](Operation *user) {
          return !isa<npuvector::TransferWriteOp, pypto::ReshapeOp>(user);
        })) {
      return rewriter.notifyMatchFailure(op, "insert_slice must feed a transfer write or reshape");
    }
    if (llvm::any_of(op.getStrides(), [](Value stride) {
          auto value = constIndexValue(stride);
          return !value || *value != 1;
        })) {
      return rewriter.notifyMatchFailure(op, "only unit strides are supported");
    }
    for (Operation *user : op->getUsers()) {
      if (auto reshape = dyn_cast<pypto::ReshapeOp>(user)) {
        rewriter.modifyOpInPlace(reshape, [&] { reshape.getSourceMutable().assign(adaptor.getSource()); });
      } else {
        cast<npuvector::TransferWriteOp>(user).getVectorMutable().assign(adaptor.getSource());
      }
    }
    rewriter.replaceOp(op, adaptor.getDest());
    return success();
  }
};

/// Drops `memref.dealloc`: the buffers become kernel arguments whose
/// lifetime the runtime manages.
class DeallocPattern : public OpConversionPattern<memref::DeallocOp> {
 public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrites `scf.parallel` into a `pypto.spmd` multi-core launch: a
/// reduction-free single induction variable sweeping `[0, N)` with step 1
/// maps onto a PyPTO SPMD grid of N cores, the induction variable being the
/// executing core's block index.
class ScfParallelPattern : public OpConversionPattern<scf::ParallelOp> {
 public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumLoops() != 1 || !op.getInitVals().empty() || op.getNumResults() != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "only reduction-free single-induction parallel loops map onto "
                                         "an SPMD grid");
    }
    auto lb = constIndexValue(op.getLowerBound()[0]);
    auto ub = constIndexValue(op.getUpperBound()[0]);
    auto step = constIndexValue(op.getStep()[0]);
    if (!lb || !ub || !step || *lb != 0 || *step != 1) {
      return rewriter.notifyMatchFailure(op, "SPMD grids sweep [0, N) with step 1");
    }
    Block &body = op.getRegion().front();
    // Even a reduction-free scf.parallel ends in an (empty) scf.reduce.
    auto reduce = dyn_cast<scf::ReduceOp>(body.getTerminator());
    if (!reduce || reduce.getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op, "reductions have no PyPTO lowering");
    }
    auto spmd = rewriter.create<pypto::SpmdOp>(op.getLoc(), rewriter.getI64IntegerAttr(*ub));
    rewriter.inlineRegionBefore(op.getRegion(), spmd.getBody(), spmd.getBody().end());
    rewriter.eraseOp(reduce);
    rewriter.replaceOp(op, spmd);
    return success();
  }
};

/// Rewrites `scf.forall` into the same `pypto.spmd` multi-core launch as
/// scf.parallel: a single induction variable sweeping `[0, N)` with step 1
/// becomes the executing core's block index. forall stores its bounds as
/// mixed static/dynamic pairs, so both spellings of a constant grid are
/// accepted. The trailing in_parallel region only wraps tensor merge
/// writes (parallel_insert_slice), which have no PyPTO counterpart, so it
/// must be empty.
class ScfForallPattern : public OpConversionPattern<scf::ForallOp> {
 public:
  using OpConversionPattern<scf::ForallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getRank() != 1 || !op.getOutputs().empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "only single-induction forall loops without shared outputs map "
                                         "onto an SPMD grid");
    }
    auto lb = mixedIndexValue(op.getMixedLowerBound()[0]);
    auto ub = mixedIndexValue(op.getMixedUpperBound()[0]);
    auto step = mixedIndexValue(op.getMixedStep()[0]);
    if (!lb || !ub || !step || *lb != 0 || *step != 1 || *ub <= 0) {
      return rewriter.notifyMatchFailure(op, "SPMD grids are non-empty sweeps [0, N) with step 1");
    }
    Block &body = op.getRegion().front();
    auto inParallel = dyn_cast<scf::InParallelOp>(body.getTerminator());
    // The in_parallel region hosts the parallel merge writes; it must hold
    // nothing but its (possibly absent) empty block.
    if (!inParallel || (!inParallel.getRegion().empty() && !inParallel.getRegion().front().empty())) {
      return rewriter.notifyMatchFailure(op, "parallel merge writes have no PyPTO lowering");
    }
    auto spmd = rewriter.create<pypto::SpmdOp>(op.getLoc(), rewriter.getI64IntegerAttr(*ub));
    rewriter.inlineRegionBefore(op.getRegion(), spmd.getBody(), spmd.getBody().end());
    rewriter.eraseOp(inParallel);
    rewriter.replaceOp(op, spmd);
    return success();
  }
};

/// Rewrites `scf.for` into `pypto.range` (which convert-pypto-to-emitpy
/// lowers to `emitpy.for`, emitted as
/// `for <iv> in pl.range(lb, ub, step)`): the per-core tile loop.
/// Loop-carried values are rejected: the generated kernels are
/// straight-line tile code with all results flowing through tile stores.
class ScfForPattern : public OpConversionPattern<scf::ForOp> {
 public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if (!op.getInitArgs().empty()) {
      return rewriter.notifyMatchFailure(op, "loop-carried values have no PyPTO lowering");
    }
    Block &body = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op, "unexpected loop terminator");
    }
    auto loop = rewriter.create<pypto::RangeOp>(op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());
    rewriter.inlineRegionBefore(op.getRegion(), loop.getBody(), loop.getBody().end());
    rewriter.eraseOp(yield);
    rewriter.replaceOp(op, loop);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function I/O rewrite (pre-conversion)
//===----------------------------------------------------------------------===//

/// Direction markers recorded by this pass and consumed by the Python
/// emitter when printing parameter annotations.
static constexpr StringLiteral kOutArgsAttr = "emitpy.out_args";
static constexpr StringLiteral kInOutArgsAttr = "emitpy.inout_args";

/// Marks a function as a PyPTO kernel; the Python emitter prints such
/// functions with an @pl.jit decorator, parameter annotations and a pl.at
/// scope around the body.
static constexpr StringLiteral kPyptoKernelAttr = "emitpy.pypto_kernel";

/// Returns the buffer a store into `value` ultimately writes when `value`
/// is a view chain (collapse_shape, subview, memory_space_cast, ...)
/// rooted in that buffer; returns `value` itself otherwise. The
/// bufferization routes stores through reshaped views of parameters, so
/// the write analysis has to look through them.
static Value stripViewChain(Value value) {
  while (Operation *defOp = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(defOp)) {
      value = view.getViewSource();
    } else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(defOp)) {
      value = castOp.getSource();
    } else {
      break;
    }
  }
  return value;
}

static void markWriteDest(Value dest, DenseSet<unsigned> &writeDests) {
  dest = stripViewChain(dest);
  if (auto arg = dyn_cast<BlockArgument>(dest)) {
    writeDests.insert(arg.getArgNumber());
  }
}

/// Returns true when `use` stores into its operand, i.e. it is the
/// destination of a transfer_write.
static bool isStoreInto(OpOperand &use) {
  Operation *owner = use.getOwner();
  if (auto write = dyn_cast<vector::TransferWriteOp>(owner)) {
    return use.get() == write.getSource();
  }
  if (auto write = dyn_cast<npuvector::TransferWriteOp>(owner)) {
    return use.get() == write.getSource();
  }
  return false;
}

static bool viewChainOnlyWritten(Value buffer);

/// Returns true when `use` does not read its operand: it either stores
/// into it, or turns it into the buffer of another view chain that is
/// itself only written.
static bool useIsStoreOrWrittenView(OpOperand &use) {
  if (isStoreInto(use)) {
    return true;
  }
  Operation *owner = use.getOwner();
  if (auto view = dyn_cast<ViewLikeOpInterface>(owner)) {
    return use.get() == view.getViewSource() && viewChainOnlyWritten(view.getOperation()->getResult(0));
  }
  if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(owner)) {
    return use.get() == castOp.getSource() && viewChainOnlyWritten(castOp.getOperation()->getResult(0));
  }
  return false;
}

/// Returns true when every use of `buffer` writes into it, directly or
/// through further view chains: such a buffer is never read.
static bool viewChainOnlyWritten(Value buffer) {
  return llvm::all_of(buffer.getUses(), [&](OpOperand &use) { return useIsStoreOrWrittenView(use); });
}

/// Returns true when `arg` is used by anything other than a store into it,
/// i.e. it is also read.
static bool argHasNonWriteUse(BlockArgument arg) {
  for (OpOperand &use : arg.getUses()) {
    if (!useIsStoreOrWrittenView(use)) {
      return true;
    }
  }
  return false;
}

/// Classifies written block arguments (step 1): arguments written by
/// transfer_write are outputs, and arguments that are additionally read
/// elsewhere become InOut instead.
static void collectWriteDestArgs(func::FuncOp funcOp, Block &entry, DenseSet<unsigned> &outArgs,
                                 DenseSet<unsigned> &inOutArgs) {
  DenseSet<unsigned> writeDests;
  funcOp.walk([&](Operation *op) {
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      markWriteDest(write.getSource(), writeDests);
    } else if (auto write = dyn_cast<npuvector::TransferWriteOp>(op)) {
      markWriteDest(write.getSource(), writeDests);
    }
  });
  for (unsigned argNumber : writeDests) {
    BlockArgument arg = entry.getArgument(argNumber);
    if (argHasNonWriteUse(arg)) {
      inOutArgs.insert(argNumber);
    } else {
      outArgs.insert(argNumber);
    }
  }
}

/// Appends output parameters for allocations that only feed stores/returns
/// (step 2): read-back allocations become InOut instead.
static void promoteAllocsToOutputs(func::FuncOp funcOp, Block &entry, DenseSet<unsigned> &outArgs,
                                   DenseSet<unsigned> &inOutArgs) {
  SmallVector<Operation *> allocs;
  funcOp.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      allocs.push_back(op);
    }
  });
  for (Operation *allocOp : allocs) {
    Value result = allocOp->getResult(0);
    bool ioOnly = true;
    bool hasRead = false;
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      if (isStoreInto(use)) {
        continue;
      }
      if (isa<func::ReturnOp, memref::DeallocOp>(owner)) {
        continue;
      }
      if (isa<vector::TransferReadOp, npuvector::TransferReadOp>(owner)) {
        hasRead = true;
        continue;
      }
      // A view chain that is only written keeps the buffer write-only;
      // any other view chain reads back through it.
      if (auto view = dyn_cast<ViewLikeOpInterface>(owner)) {
        if (use.get() == view.getViewSource()) {
          if (viewChainOnlyWritten(view.getOperation()->getResult(0))) {
            continue;
          }
          hasRead = true;
          continue;
        }
      }
      if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(owner)) {
        if (use.get() == castOp.getSource()) {
          if (viewChainOnlyWritten(castOp.getOperation()->getResult(0))) {
            continue;
          }
          hasRead = true;
          continue;
        }
      }
      ioOnly = false;
      break;
    }
    if (!ioOnly) {
      continue;  // leaves the op for the conversion to reject
    }

    BlockArgument newArg = entry.addArgument(cast<MemRefType>(result.getType()), allocOp->getLoc());
    if (hasRead) {
      inOutArgs.insert(newArg.getArgNumber());
    } else {
      outArgs.insert(newArg.getArgNumber());
    }
    result.replaceAllUsesWith(newArg);
    allocOp->erase();
  }
}

/// Appends an output tensor for a computed tile return value and stores
/// the tile into it.
static LogicalResult storeTileReturn(OpBuilder &builder, Block &entry, DenseSet<unsigned> &outArgs, func::ReturnOp ret,
                                     Value operand, ShapedType shaped) {
  auto memrefType = MemRefType::get(shaped.getShape(), shaped.getElementType());
  BlockArgument newArg = entry.addArgument(memrefType, ret.getLoc());
  SmallVector<Value> zeros;
  for (int64_t i = 0; i < shaped.getRank(); ++i) {
    zeros.push_back(builder.create<arith::ConstantIndexOp>(ret.getLoc(), 0));
  }
  if (isa<npuvector::NPUVectorType>(operand.getType())) {
    builder.create<npuvector::TransferWriteOp>(ret.getLoc(), TypeRange{}, operand, newArg, zeros,
                                               /*mask=*/Value());
  } else {
    builder.create<vector::TransferWriteOp>(ret.getLoc(), operand, newArg, zeros);
  }
  outArgs.insert(newArg.getArgNumber());
  return success();
}

/// Appends an output tensor for a computed scalar return value: the scalar
/// is broadcast to vector<1> and stored.
static void storeScalarReturn(OpBuilder &builder, Block &entry, DenseSet<unsigned> &outArgs, func::ReturnOp ret,
                              Value operand) {
  Type type = operand.getType();
  auto vecType = VectorType::get({1}, type);
  auto memrefType = MemRefType::get({1}, type);
  BlockArgument newArg = entry.addArgument(memrefType, ret.getLoc());
  Value broadcast = builder.create<vector::BroadcastOp>(ret.getLoc(), vecType, operand);
  Value zero = builder.create<arith::ConstantIndexOp>(ret.getLoc(), 0);
  SmallVector<Value> storeIndices{zero};
  builder.create<vector::TransferWriteOp>(ret.getLoc(), broadcast, newArg, storeIndices);
  outArgs.insert(newArg.getArgNumber());
}

/// Rewrites every `return` (step 3): computed values are stored into
/// appended output parameters and the return itself is emptied. PyPTO
/// kernels return their outputs by name; the Python emitter reconstructs
/// that from the recorded directions, so the IR keeps a plain `return`.
static LogicalResult rewriteReturns(OpBuilder &builder, Block &entry, ArrayRef<func::ReturnOp> returns,
                                    DenseSet<unsigned> &outArgs, DenseSet<unsigned> &inOutArgs) {
  for (func::ReturnOp ret : returns) {
    builder.setInsertionPoint(ret);
    for (Value operand : ret.getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        // Returning an argument that is not an output still needs a
        // direction that permits both read and write.
        if (!outArgs.count(arg.getArgNumber())) {
          inOutArgs.insert(arg.getArgNumber());
        }
        continue;
      }
      Type type = operand.getType();
      auto shaped = dyn_cast<ShapedType>(type);
      if (shaped && shaped.hasStaticShape() && (isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type))) {
        if (failed(storeTileReturn(builder, entry, outArgs, ret, operand, shaped))) {
          return failure();
        }
        continue;
      }
      if (isa<FloatType>(type) || isa<IntegerType>(type) || isa<IndexType>(type)) {
        storeScalarReturn(builder, entry, outArgs, ret, operand);
        continue;
      }
      return ret.emitOpError("unsupported return value type for PyPTO");
    }
    ret.getOperandsMutable().clear();
  }
  return success();
}

/// Records the Out/InOut parameter directions for the emitter (step 5).
/// DenseI64ArrayAttr keeps its identity across the textual round-trip
/// (`array<i64: ...>`), unlike the generic ArrayAttr the emitter would not
/// find again.
static void recordArgDirections(func::FuncOp funcOp, const DenseSet<unsigned> &outArgs,
                                const DenseSet<unsigned> &inOutArgs) {
  MLIRContext *ctx = funcOp.getContext();
  if (!outArgs.empty()) {
    SmallVector<int64_t> outList(outArgs.begin(), outArgs.end());
    llvm::sort(outList);
    funcOp->setAttr(kOutArgsAttr, DenseI64ArrayAttr::get(ctx, outList));
  }
  if (!inOutArgs.empty()) {
    SmallVector<int64_t> inOutList(inOutArgs.begin(), inOutArgs.end());
    llvm::sort(inOutList);
    funcOp->setAttr(kInOutArgsAttr, DenseI64ArrayAttr::get(ctx, inOutList));
  }
}

/// Rewrites the function so that every output flows through a `pl.Out`
/// parameter, mirroring the PyPTO kernel model:
/// - arguments written by `transfer_write` become outputs;
/// - allocations that only feed stores/returns become appended output
///   arguments (read-back allocations become `pl.InOut` instead);
/// - computed return values are stored into appended output arguments
///   (scalars through a broadcast to vector<1>);
/// - every `return` carries the output arguments, since a bare `return` is
///   not a legal PyPTO kernel tail.
static LogicalResult rewriteFunctionOutputs(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  Block &entry = funcOp.getBody().front();

  DenseSet<unsigned> outArgs, inOutArgs;

  // 1. Arguments written by transfer_write are outputs; arguments that are
  // additionally read elsewhere become InOut instead.
  collectWriteDestArgs(funcOp, entry, outArgs, inOutArgs);

  // 2. Allocations that only feed stores/returns become output arguments.
  promoteAllocsToOutputs(funcOp, entry, outArgs, inOutArgs);

  // 3. Rewrite returns: computed values are stored into appended output
  // parameters and the return itself is emptied.
  SmallVector<func::ReturnOp> returns;
  funcOp.walk([&](func::ReturnOp op) { returns.push_back(op); });
  if (failed(rewriteReturns(builder, entry, returns, outArgs, inOutArgs))) {
    return failure();
  }

  // 4. The kernel hands results back through its Out parameters only. The
  // input list is rebuilt from the block arguments: appended output
  // parameters are only visible there, not in the old function type.
  if (!returns.empty()) {
    SmallVector<Type> inputTypes;
    llvm::transform(entry.getArguments(), std::back_inserter(inputTypes),
                    [](BlockArgument arg) { return arg.getType(); });
    funcOp.setType(FunctionType::get(ctx, inputTypes, {}));
  }

  // 5. Record the directions for the emitter.
  recordArgDirections(funcOp, outArgs, inOutArgs);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Makes sure the module enclosing `funcOp` imports pypto.language as `pl`
/// exactly once, so the generated callees (pl.*, ...) resolve.
static void insertPyptoImport(func::FuncOp funcOp) {
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  if (!module) {
    return;
  }
  // Function passes run in parallel on sibling functions; the module level
  // import bookkeeping (lookup plus insertion) must be serialized.
  static std::mutex importMutex;
  std::lock_guard<std::mutex> lock(importMutex);
  for (Operation &op : module.getBody()->getOperations()) {
    auto importOp = dyn_cast<emitpy::ImportOp>(op);
    if (!importOp) {
      continue;
    }
    auto alias = importOp.getModuleAlias();
    if (importOp.getModuleName() == "pypto.language" && alias && *alias == "pl") {
      return;
    }
  }
  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<emitpy::ImportOp>(module.getLoc(), "pypto.language", builder.getStringAttr("pl"),
                                   /*members_to_import=*/nullptr,
                                   /*member_aliases=*/nullptr,
                                   /*import_all=*/nullptr);
}

/// Returns true for statically shaped types holding at most one element
/// (0-D or size-1 tiles).
static bool isScalarTileType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasStaticShape() && (shaped.getRank() == 0 || shaped.getNumElements() == 1);
}

/// Undoes the size-1 shape folds (memref.collapse_shape) that the shape
/// normalization inserts for scalar tensors: a 0-D buffer holds the same
/// single element as the [1] buffer it came from, so every use is reroute
/// to the original buffer and the 0-D reads/writes keep addressing it at
/// [0]. Doing this before the signature rewrite keeps the parameters and
/// the output analysis on the original buffers.
static void undoSizeOneShapeFolds(func::FuncOp funcOp) {
  SmallVector<memref::CollapseShapeOp> folds;
  funcOp.walk([&](memref::CollapseShapeOp fold) {
    auto srcType = dyn_cast<ShapedType>(fold.getSrc().getType());
    auto dstType = dyn_cast<ShapedType>(fold.getResult().getType());
    if (srcType && dstType && srcType.getNumElements() == 1 && dstType.getNumElements() == 1) {
      folds.push_back(fold);
    }
  });
  for (memref::CollapseShapeOp fold : folds) {
    Value src = fold.getSrc();
    unsigned srcRank = cast<ShapedType>(src.getType()).getRank();
    // Every 0-D access (no coordinates) of the folded buffer must be
    // re-addressed at the zero coordinate of the unfolded one, one
    // zero per axis.
    if (srcRank > 0) {
      SmallVector<Operation *> users(fold.getResult().getUsers());
      for (Operation *user : users) {
        auto fixIndices = [&](MutableOperandRange indices) {
          if (indices.size() != 0) {
            return;
          }
          OpBuilder builder(user);
          SmallVector<Value> coords;
          for (unsigned i = 0; i < srcRank; ++i) {
            coords.push_back(builder.create<arith::ConstantIndexOp>(fold.getLoc(), 0));
          }
          indices.assign(coords);
        };
        if (auto read = dyn_cast<npuvector::TransferReadOp>(user)) {
          fixIndices(read.getIndicesMutable());
        } else if (auto write = dyn_cast<npuvector::TransferWriteOp>(user)) {
          fixIndices(write.getIndicesMutable());
        }
      }
    }
    fold.getResult().replaceAllUsesWith(src);
    fold.erase();
  }
}

/// Lowers the 0-D (scalar) chains onto plain scalars before the dialect
/// conversion: the shape normalization folds size-1 tensors to 0-D tiles,
/// which have no device form. A 0-D read becomes pypto.read of the single
/// element, a 0-D broadcast is its source itself, any other op's results
/// adopt the scalar element type, and 0-D stores are left for the
/// transfer-write pattern (which writes scalars through pypto.write). Doing
/// this ahead of the conversion keeps the value types stable while the
/// patterns rewrite ops one by one.
static void lowerZeroDimChains(func::FuncOp funcOp) {
  SmallVector<Operation *> scalarOps;
  funcOp.walk([&](Operation *op) {
    if (llvm::any_of(op->getResultTypes(), isScalarTileType)) {
      scalarOps.push_back(op);
    }
  });
  OpBuilder builder(funcOp);
  for (Operation *op : scalarOps) {
    if (isa<npuvector::TransferReadOp, vector::TransferReadOp>(op)) {
      // %v = pypto.read src[...]: read the single element as a scalar,
      // forwarding the transfer's own coordinates. A genuinely 0-D source
      // (a scalar function argument) addresses no dimension at all; the
      // unfolded [1] buffer keeps its (possibly runtime) coordinate.
      Value source = op->getOperand(0);
      auto shaped = cast<ShapedType>(op->getResult(0).getType());
      ValueRange indices = TypeSwitch<Operation *, ValueRange>(op)
                             .Case<npuvector::TransferReadOp>(
                               [](npuvector::TransferReadOp read) { return read.getIndices(); })
                             .Case<vector::TransferReadOp>(
                               [](vector::TransferReadOp read) { return read.getIndices(); });
      builder.setInsertionPoint(op);
      Value read =
        builder.create<pypto::ReadOp>(op->getLoc(), shaped.getElementType(), source, indices).getResult();
      op->getResult(0).replaceAllUsesWith(read);
      op->erase();
      continue;
    }
    if (isa<npuvector::BroadcastOp, vector::BroadcastOp>(op)) {
      // A 0-D broadcast of a scalar is the scalar itself.
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
      continue;
    }
    for (OpResult res : op->getResults()) {
      auto shaped = dyn_cast<ShapedType>(res.getType());
      if (!shaped || !shaped.hasStaticShape() || shaped.getNumElements() != 1) {
        continue;
      }
      if (shaped.getRank() != 0) {
        // A size-1 tile is a scalar only when the chain under it was
        // fully lowered: reads and broadcasts above already replaced
        // their results with scalars, so an elementwise arith/math op
        // whose operands are all scalars is such a chain's tail -- and
        // an arith/math op with scalar operands and a tile result is
        // otherwise never well-typed. npuvector comparisons and selects
        // are the same kind of tail: their operands and result must
        // share a shape, so all-scalar operands naming a size-1 tile
        // result is again only a lowered chain's tail. Retype the
        // result to the element type to keep the op verifiable;
        // anything else may still be a genuine tile and keeps its
        // shape.
        StringRef ns = op->getDialect()->getNamespace();
        bool scalarChainTail =
          op->getNumOperands() > 0 &&
          llvm::all_of(op->getOperands(), [](Value operand) { return !isa<ShapedType>(operand.getType()); }) &&
          ((ns == "arith" || ns == "math") ||
           isa<npuvector::CmpFOp, npuvector::CmpIOp, npuvector::SelectOp>(op));
        if (!scalarChainTail) {
          continue;
        }
      }
      res.setType(shaped.getElementType());
    }
  }
}

/// One fold step: rewrites every `pypto.load`/`pypto.store` reading through
/// `slice` to address the slice's base directly, with the window offsets
/// added onto the memory op's own offsets. Returns whether anything changed.
///
/// The runtime materializes a slice view with strides derived densely from
/// the window shape, so a window narrower than the source row loses the
/// source's row stride. Folding the window offsets into the memory op and
/// addressing the base keeps the source coordinates -- and with them the
/// source's true strides -- intact.
static bool foldSliceIntoMemoryOps(pypto::SliceOp slice) {
  // Rank-reducing views keep their own slice op: the window addresses fewer
  // coordinates than the source has axes.
  if (slice.getDropDims()) {
    return false;
  }
  auto srcType = dyn_cast<ShapedType>(slice.getSource().getType());
  if (!srcType || slice.getOffsets().size() != static_cast<size_t>(srcType.getRank())) {
    return false;
  }
  auto memOffsets = [](Operation *op) -> Operation::operand_range {
    if (auto load = dyn_cast<pypto::LoadOp>(op)) {
      return load.getOffsets();
    }
    return cast<pypto::StoreOp>(op).getOffsets();
  };
  // The fold only applies when every consumer is a memory op addressing one
  // coordinate per source axis; anything else (another view, a tile op)
  // keeps the slice.
  SmallVector<Operation *> users(slice->getUsers());
  if (users.empty()) {
    slice->erase();
    return true;
  }
  for (Operation *user : users) {
    if (!isa<pypto::LoadOp, pypto::StoreOp>(user) ||
        memOffsets(user).size() != static_cast<size_t>(srcType.getRank())) {
      return false;
    }
  }
  OpBuilder builder(slice);
  for (Operation *user : users) {
    builder.setInsertionPoint(user);
    SmallVector<Value> folded;
    folded.reserve(memOffsets(user).size());
    for (auto [offset, windowOffset] : llvm::zip(memOffsets(user), slice.getOffsets())) {
      auto windowCst = constIndexValue(windowOffset);
      if (windowCst && *windowCst == 0) {
        folded.push_back(offset);
        continue;
      }
      auto offsetCst = constIndexValue(offset);
      if (windowCst && offsetCst) {
        folded.push_back(builder.create<arith::ConstantIndexOp>(slice.getLoc(), *offsetCst + *windowCst));
        continue;
      }
      // Dynamic offsets ride on scalar arith.addi, which stage two maps
      // onto the infix addi pseudo-callee.
      folded.push_back(builder.create<arith::AddIOp>(slice.getLoc(), builder.getIndexType(), offset, windowOffset));
    }
    // load operands are [source, offsets..., padded..., valid...] and store
    // operands [value, dest, offsets...]; the base sits one slot earlier on
    // a load.
    unsigned base = isa<pypto::LoadOp>(user) ? 0 : 1;
    user->setOperand(base, slice.getSource());
    for (unsigned i = 0, e = folded.size(); i < e; ++i) {
      user->setOperand(base + 1 + i, folded[i]);
    }
  }
  if (slice->use_empty()) {
    slice->erase();
  }
  return true;
}

/// Folds slice views into their memory-op consumers until nothing changes,
/// so chains of views (a slice reading another slice) collapse one level per
/// round.
static void foldSliceViews(func::FuncOp funcOp) {
  while (true) {
    SmallVector<pypto::SliceOp> worklist;
    funcOp->walk([&](pypto::SliceOp op) { worklist.push_back(op); });
    // any_of short-circuits after the first successful fold; the outer loop
    // then restarts the round, which keeps the fixpoint semantics unchanged.
    if (!llvm::any_of(worklist, [](pypto::SliceOp slice) { return foldSliceIntoMemoryOps(slice); })) {
      break;
    }
  }
}

/// The pypto DSL lowers `with pl.spmd(N):` into a standalone InCore
/// function and rejects tile-typed captures (`InCore parameters must be
/// TensorType`), so a `pypto.tile_full` defined outside the region but
/// used only inside it must be created inside it. A `pypto.constant`
/// crosses the boundary as a Var instead, which loses the compile-time
/// constant property `pl.tile.select` scalar branches require — so it too
/// must be created inside. Sinking is always dominance-safe: the op moves
/// down past its position (operands defined before the region still
/// dominate it) and lands before the earliest user. Users may live in
/// sibling nested regions (e.g. two `pypto.range` loops in the spmd body),
/// so they cannot be compared directly — instead each user is mapped to its
/// top-level ancestor op inside the spmd entry block, and the earliest
/// ancestor is located by a linear scan of that (tiny) block. This avoids
/// `isBeforeInBlock`, whose `orderIndex` update path dereferences null
/// prev/next nodes on single-op blocks holding freshly created ops.
static void sinkConstantsIntoSpmd(func::FuncOp funcOp) {
  for (pypto::SpmdOp spmd : llvm::to_vector(funcOp.getOps<pypto::SpmdOp>())) {
    Block *parent = spmd->getBlock();
    Block &entry = spmd.getBody().front();
    // Only ops preceding the spmd can be used inside its region (anything
    // after it has no dominance over the region body).
    for (Operation &op : llvm::make_early_inc_range(*parent)) {
      if (&op == spmd.getOperation()) {
        break;
      }
      if (!isa<pypto::TileFullOp, pypto::ConstantOp>(op) || op.use_empty()) {
        continue;
      }
      llvm::SmallPtrSet<Operation *, 8> ancestors;
      bool allInside = true;
      for (Operation *user : op.getUsers()) {
        Operation *ancestor = entry.findAncestorOpInBlock(*user);
        if (!ancestor) {
          allInside = false;
          break;
        }
        ancestors.insert(ancestor);
      }
      if (!allInside || ancestors.empty()) {
        continue;
      }
      for (Operation &entryOp : entry) {
        if (ancestors.count(&entryOp)) {
          op.moveBefore(&entryOp);
          break;
        }
      }
    }
  }
}

/// A `pypto.read` of a bf16 element cannot live at the orchestration
/// level: the emitted C++ materializes the scalar as `bfloat16_t`, which
/// the host compiler can neither `static_cast` to float nor pass through
/// the runtime `add_scalar` (it rejects non-arithmetic C++ types). When
/// such a read only feeds -- through widening `pypto.cast`s -- uses inside
/// an spmd region, the whole chain belongs inside that region instead:
/// the bf16 tensor then crosses as a plain kernel input (`add_input` has
/// no dtype restriction) and the read/cast run in the kernel, where they
/// lower to native MLIR bf16 ops (`pto.load` / `arith.extf`).
static void sinkBf16ReadCastsIntoSpmd(func::FuncOp funcOp) {
  for (pypto::SpmdOp spmd : llvm::to_vector(funcOp.getOps<pypto::SpmdOp>())) {
    Block *parent = spmd->getBlock();
    Block &entry = spmd.getBody().front();
    // Candidates are collected first: the moves below unlink ops from
    // this block, which would invalidate a live iterator over it.
    SmallVector<pypto::ReadOp> reads;
    for (Operation &op : *parent) {
      if (&op == spmd.getOperation()) {
        break;
      }
      auto read = dyn_cast<pypto::ReadOp>(op);
      if (read && read.getResult().getType().isBF16()) {
        reads.push_back(read);
      }
    }
    for (pypto::ReadOp read : reads) {
      // Walk the widening cast chain while it stays in the parent block;
      // the chain tail is the value whose uses decide the sink.
      SmallVector<Operation *> chain;
      Value value = read.getResult();
      while (value.hasOneUse()) {
        Operation *user = *value.getUsers().begin();
        auto cast = dyn_cast<pypto::CastOp>(user);
        if (!cast || cast->getBlock() != parent) {
          break;
        }
        chain.push_back(cast);
        value = cast.getResult();
      }
      // Every use of the chain tail must live inside the spmd region;
      // like above, users may be nested, so they are mapped to their
      // top-level ancestors in the entry block.
      llvm::SmallPtrSet<Operation *, 8> ancestors;
      bool allInside = !value.use_empty();
      for (Operation *user : value.getUsers()) {
        Operation *ancestor = entry.findAncestorOpInBlock(*user);
        if (!ancestor) {
          allInside = false;
          break;
        }
        ancestors.insert(ancestor);
      }
      if (!allInside) {
        continue;
      }
      // The earliest use inside the entry block anchors the insertion.
      Operation *target = nullptr;
      for (Operation &entryOp : entry) {
        if (ancestors.count(&entryOp)) {
          target = &entryOp;
          break;
        }
      }
      if (!target) {
        continue;
      }
      // Move the chain tail first so each op lands right before its
      // consumer, with the read ending up before the first cast.
      Operation *anchor = target;
      for (Operation *op : llvm::reverse(chain)) {
        op->moveBefore(anchor);
        anchor = op;
      }
      read->moveBefore(anchor);
    }
  }
}

/// Promotes 0-D (scalar) output parameters to [1]: the device runtime has
/// no 0-D tensor form, so a scalar output must be handed out through a
/// one-element buffer -- the same shape `storeScalarReturn` appends for
/// computed scalar returns. The Python emitter rebuilds the kernel
/// signature from the function type, so the parameter itself carries the
/// [1] shape, and every access of the promoted buffer is re-addressed at
/// [0].
static void promoteZeroDimOutArgs(func::FuncOp funcOp) {
  Block &entry = funcOp.getBody().front();
  DenseSet<unsigned> outArgs;
  for (StringRef attrName : {kOutArgsAttr, kInOutArgsAttr}) {
    if (auto list = funcOp->getAttrOfType<DenseI64ArrayAttr>(attrName)) {
      for (int64_t argNo : list.asArrayRef()) {
        outArgs.insert(static_cast<unsigned>(argNo));
      }
    }
  }
  SmallVector<BlockArgument> candidates;
  for (BlockArgument arg : entry.getArguments()) {
    auto memrefType = dyn_cast<MemRefType>(arg.getType());
    if (!memrefType || memrefType.getRank() != 0 || !outArgs.count(arg.getArgNumber())) {
      continue;
    }
    // Only plain element accesses may be re-addressed; a view of the
    // buffer keeps the scalar parameter as it came in.
    bool promotable = true;
    for (OpOperand &use : arg.getUses()) {
      if (!isa<pypto::ReadOp, pypto::WriteOp>(use.getOwner())) {
        promotable = false;
        break;
      }
    }
    if (promotable) {
      candidates.push_back(arg);
    }
  }
  if (candidates.empty()) {
    return;
  }
  for (BlockArgument arg : candidates) {
    auto memrefType = cast<MemRefType>(arg.getType());
    arg.setType(MemRefType::get({1}, memrefType.getElementType()));
    for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
      Operation *access = use.getOwner();
      OpBuilder builder(access);
      Value zero = builder.create<arith::ConstantIndexOp>(access->getLoc(), 0);
      if (auto read = dyn_cast<pypto::ReadOp>(access)) {
        if (read.getIndices().empty()) {
          read.getIndicesMutable().append(zero);
        }
      } else {
        auto write = cast<pypto::WriteOp>(access);
        if (write.getOffsets().empty()) {
          write.getOffsetsMutable().append(zero);
        }
      }
    }
  }
  MLIRContext *ctx = funcOp.getContext();
  SmallVector<Type> inputTypes;
  llvm::transform(entry.getArguments(), std::back_inserter(inputTypes),
                  [](BlockArgument arg) { return arg.getType(); });
  funcOp.setType(FunctionType::get(ctx, inputTypes, funcOp.getFunctionType().getResults()));
}

struct ConvertVectorToPyptoPass : impl::ConvertVectorToPyptoBase<ConvertVectorToPyptoPass> {
  using impl::ConvertVectorToPyptoBase<ConvertVectorToPyptoPass>::ConvertVectorToPyptoBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect, pypto::PyptoDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    insertPyptoImport(funcOp);

    // Mark the function so the Python emitter renders it as a @pl.jit
    // kernel rather than a plain function.
    funcOp->setAttr(kPyptoKernelAttr, UnitAttr::get(ctx));

    undoSizeOneShapeFolds(funcOp);

    // Normalize the function I/O to the PyPTO Out-parameter model before
    // converting: the rewrite inserts npuvector/vector transfer ops that the
    // patterns below then lower.
    if (failed(rewriteFunctionOutputs(funcOp))) {
      signalPassFailure();
      return;
    }

    lowerZeroDimChains(funcOp);

    VectorToPyptoTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    // The patterns emit typed pypto ops that keep the source result types;
    // pypto is the output dialect of this pass, and the downstream
    // convert-pypto-to-emitpy pass lowers it to the emitpy call forms.
    // Nothing in this stage may produce emitpy ops.
    target.addLegalDialect<pypto::PyptoDialect>();
    target
      .addIllegalDialect<npuvector::NPUVectorDialect, vector::VectorDialect, arith::ArithDialect, math::MathDialect>();
    // Scalar arithmetic keeps the arith op: pypto is a tile dialect, so
    // scalar adds/subs/muls/divs, compares, selects, the floordivsi of the
    // affine floordiv chain, the scalar integer remainders (`pl.fmod` is
    // tile-only) and the index constants feeding `pypto.range` bounds all
    // wait for stage two, which maps them onto infix pseudo-callees named
    // after the arith ops (`emitpy.literal` for the constants).
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) { return isa<IndexType>(op.getType()); });
    target.addDynamicallyLegalOp<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp, arith::AddFOp,
                                 arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::CmpIOp, arith::CmpFOp,
                                 arith::SelectOp, arith::FloorDivSIOp, arith::RemSIOp, arith::RemUIOp,
                                 arith::NegFOp>(
      [](Operation *op) { return !isa<VectorType, npuvector::NPUVectorType>(op->getOperand(0).getType()); });
    // Scalar float math keeps the math op the same way: `pl.sqrt`/`pl.rsqrt`
    // are Tensor/Tile-only APIs, so scalar square roots (e.g. the batch-norm
    // scale computation) wait for stage two, which maps them onto
    // pseudo-callees the translator lowers through a small tile detour
    // (the scalar `** 0.5` power form is numerically wrong on the backend).
    target.addDynamicallyLegalOp<math::SqrtOp, math::RsqrtOp, math::ExpOp, math::LogOp>(
      [](Operation *op) { return !isa<VectorType, npuvector::NPUVectorType>(op->getOperand(0).getType()); });
    // Control flow becomes pypto here: scf.parallel/scf.forall map onto
    // `pypto.spmd` multi-core launches and scf.for onto the `pypto.range`
    // per-core tile loop.
    target.addIllegalDialect<scf::SCFDialect>();
    // Allocations are handled by rewriteFunctionOutputs; any survivor means
    // a use this backend cannot express. The view ops must become `pl.*`
    // tiles: the Python emitter has no memref form, so leaving them legal
    // lets them survive the conversion as raw memrefs.
    target.addIllegalOp<memref::AllocOp, memref::AllocaOp, memref::DeallocOp, memref::LoadOp, memref::StoreOp,
                        memref::CollapseShapeOp, memref::ExpandShapeOp, memref::TransposeOp, memref::ReshapeOp,
                        memref::SubViewOp, memref::ReinterpretCastOp>();
    // The signature itself stays untouched by the conversion: retyping
    // parameters through the framework would materialize
    // unrealized_conversion_cast ops that a full conversion rejects. The
    // signature and block arguments are rewritten by hand in the downstream
    // convert-pypto-to-emitpy pass, once the body is fully converted.
    // Returns were emptied by the pre-pass rewrite and calls are left as-is.
    target.addLegalOp<func::FuncOp, func::ReturnOp, func::CallOp>();

    RewritePatternSet patterns(ctx);
    vector_to_pypto::populateVectorToPyptoConversionPatterns(patterns, typeConverter);

    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Promote 0-D output parameters to [1] before the slice folding: the
    // re-addressed [0] accesses must be in place before any view folding
    // could observe the buffer.
    promoteZeroDimOutArgs(funcOp);

    // Fold slice views into their memory-op consumers: the runtime derives a
    // view's strides densely from the window shape, losing the source's row
    // stride whenever the window is narrower than the source row, so slices
    // that only feed loads/stores address the base directly instead.
    foldSliceViews(funcOp);

    // Sink bf16 scalar reads that only feed an spmd body through their
    // widening casts into that body, ahead of the constants pass so index
    // constants feeding a sunk read follow it inside and stay literals:
    // the host emitter materializes such a read as a `bfloat16_t` C++
    // scalar that the host compiler cannot work with, while the kernel
    // reads it from a plain tensor input and casts it natively.
    sinkBf16ReadCastsIntoSpmd(funcOp);

    // Sink tile constants that only feed an spmd body into that body: the
    // DSL lowers `with pl.spmd(N):` into a standalone InCore function that
    // cannot capture tile values; scalar constants would cross as runtime
    // Vars, losing the compile-time constant property.
    sinkConstantsIntoSpmd(funcOp);
  }
};

}  // namespace

namespace vector_to_pypto {

namespace {

/// Constants and elementwise float arithmetic.
static void populateFloatArithPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Constants.
  patterns.add<ArithConstantPattern>(typeConverter, ctx);

  // Elementwise float arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddFOp, pypto::AddOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubFOp, pypto::SubOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulFOp, pypto::MulOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivFOp, pypto::DivOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::RemFOp, pypto::FmodOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MaximumFOp, pypto::MaximumOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinimumFOp, pypto::MinimumOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::NegFOp, pypto::NegOp>>(typeConverter, ctx);
}

/// Elementwise integer arithmetic.
static void populateIntArithPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Elementwise integer arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddIOp, pypto::AddOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubIOp, pypto::SubOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulIOp, pypto::MulOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivSIOp, pypto::DivOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivUIOp, pypto::DivOp>>(typeConverter, ctx);
  patterns.add<TileRemToPyptoPattern<arith::RemSIOp>>(typeConverter, ctx);
  patterns.add<TileRemToPyptoPattern<arith::RemUIOp>>(typeConverter, ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MaxSIOp, pypto::MaxOp, pypto::MaximumOp>>(typeConverter, ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MinSIOp, pypto::MinOp, pypto::MinimumOp>>(typeConverter, ctx);
  // Floor division stays arith.floordivsi here: the translator prints the
  // `floordivsi` pseudo-callee it lowers to as an infix `(a // b)`
  // expression, and convert-pypto-to-emitpy introduces that call.
  patterns.add<OpToPyptoOpPattern<arith::MaxUIOp, pypto::MaximumOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinUIOp, pypto::MinimumOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::AndIOp, pypto::AndOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::OrIOp, pypto::OrOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::XOrIOp, pypto::XorOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShLIOp, pypto::ShlOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRSIOp, pypto::ShrOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRUIOp, pypto::ShrOp>>(typeConverter, ctx);
}

/// Comparisons and select.
static void populateCmpSelectPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Comparisons and select.
  patterns.add<CmpToPyptoPattern<arith::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<arith::CmpFOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpFOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<arith::SelectOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<npuvector::SelectOp>>(typeConverter, ctx);
}

/// Numeric casts (pl.cast) and bitcasts (pl.reinterpret_view).
static void populateCastPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Numeric casts -> pl.cast; bitcasts -> pl.reinterpret_view.
  patterns.add<CastToPyptoPattern<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::ExtSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::ExtUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::TruncIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::UIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::FPToUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::SIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::UIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastUIOp>>(typeConverter, ctx);
  patterns.add<BitcastToPyptoPattern<arith::BitcastOp>>(typeConverter, ctx);
  patterns.add<BitcastToPyptoPattern<npuvector::BitcastOp>>(typeConverter, ctx);
}

/// Rewrites `math.round`/`math.floor`/`math.ceil` on a tile into the PyPTO
/// round-trip cast chain: `pl.cast` rounds float-to-int with the named
/// rounding mode, and the widening cast back restores the source dtype.
/// PyPTO has no first-class rounding op.
template <typename OpTy>
class RoundingCastPattern : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  RoundingCastPattern(TypeConverter &typeConverter, MLIRContext *ctx, StringRef mode)
      : OpConversionPattern<OpTy>(typeConverter, ctx), mode(mode.str()) {}

  LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto shaped = dyn_cast<ShapedType>(op.getOperand().getType());
    if (!shaped) {
      return rewriter.notifyMatchFailure(op, "rounding input must be a tile");
    }
    if (!isa<FloatType>(shaped.getElementType())) {
      return rewriter.notifyMatchFailure(op, "rounding input must hold float elements");
    }
    Location loc = op.getLoc();
    Type intTile = pypto::TileType::get(rewriter.getContext(), rewriter.getIntegerType(32), shaped.getShape());
    Value asInt = rewriter.create<pypto::CastOp>(loc, intTile, op.getOperand());
    asInt.getDefiningOp<pypto::CastOp>()->setAttr("mode", rewriter.getStringAttr(mode));
    rewriter.replaceOpWithNewOp<pypto::CastOp>(op, op->getResultTypes(), asInt);
    return success();
  }

 private:
  std::string mode;
};

/// Math dialect elementwise functions.
static void populateMathPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Math dialect elementwise functions with PyPTO counterparts.
  patterns.add<OpToPyptoOpPattern<math::AbsFOp, pypto::AbsOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::AbsIOp, pypto::AbsOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::ExpOp, pypto::ExpOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::LogOp, pypto::LogOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::SqrtOp, pypto::SqrtOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::RsqrtOp, pypto::RsqrtOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::SinOp, pypto::SinOp>>(typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::CosOp, pypto::CosOp>>(typeConverter, ctx);
  patterns.add<RoundingCastPattern<math::RoundOp>>(typeConverter, ctx, "round");
  patterns.add<RoundingCastPattern<math::FloorOp>>(typeConverter, ctx, "floor");
  patterns.add<RoundingCastPattern<math::CeilOp>>(typeConverter, ctx, "ceil");
}

/// Tile ops, memref management, and control flow.
static void populateTileAndControlFlowPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // npuvector / vector tile ops.
  patterns.add<TransferReadPattern<npuvector::TransferReadOp>>(typeConverter, ctx);
  patterns.add<TransferReadPattern<vector::TransferReadOp>>(typeConverter, ctx);
  patterns.add<TransferWritePattern<npuvector::TransferWriteOp>>(typeConverter, ctx);
  patterns.add<TransferWritePattern<vector::TransferWriteOp>>(typeConverter, ctx);
  patterns.add<ExtractSlicePattern>(typeConverter, ctx);
  patterns.add<InsertSlicePattern>(typeConverter, ctx);
  patterns.add<ReductionPattern<npuvector::ReductionOp>>(typeConverter, ctx);
  patterns.add<ReductionPattern<vector::ReductionOp>>(typeConverter, ctx);
  // Comparison-mask lift before both broadcast fusions: a rank-1 `i1`
  // comparison lifted to rank-2 rebuilds `pypto.cmp` at the broadcast's
  // geometry (pl.cmp never broadcasts and pl.tile.select demands a
  // full-geometry condition). The benefit outranks the tile-broadcast
  // fusion so an i1 source cannot stray onto the arithmetic expand
  // family, and the always-matching fallback.
  patterns.add<BroadcastCmpMaskPattern>(typeConverter, ctx, PatternBenefit(3));
  // Tile-broadcast fusion before the scalar-broadcast fallback: a rank-1
  // tile lifted to rank-2 by `dimension = [0]/[1]` and consumed by arith
  // add/sub/mul/div folds into the runtime-native row/col_expand family,
  // fusing every consumer. The benefit outranks BroadcastPattern so the
  // always-matching fallback cannot starve the fusion.
  patterns.add<TileBroadcastFusionPattern>(typeConverter, ctx, PatternBenefit(2));
  // Scalar-broadcast fusion before the scalar-broadcast fallback: a
  // scalar consumed by comparisons and select branches rides through as
  // the tile-scalar `pl.tile.cmps` / `pl.tile.select` forms. The fusion
  // carries a higher benefit than BroadcastPattern: with equal benefits
  // the pattern order is unspecified, and the always-matching fallback
  // would otherwise starve the fusion.
  patterns.add<ScalarBroadcastFusionPattern>(typeConverter, ctx, PatternBenefit(2));
  patterns.add<BroadcastPattern<npuvector::BroadcastOp>>(typeConverter, ctx);
  patterns.add<BroadcastPattern<vector::BroadcastOp>>(typeConverter, ctx);
  patterns.add<TransposePattern<npuvector::TransposeOp>>(typeConverter, ctx);
  patterns.add<TransposePattern<vector::TransposeOp>>(typeConverter, ctx);

  // memref management ops.
  patterns.add<CollapseShapePattern>(typeConverter, ctx);
  patterns.add<ExpandShapePattern>(typeConverter, ctx);
  patterns.add<MemrefTransposePattern>(typeConverter, ctx);
  patterns.add<ReshapePattern>(typeConverter, ctx);
  patterns.add<SubviewPattern>(typeConverter, ctx);
  patterns.add<ReinterpretCastPattern>(typeConverter, ctx);
  patterns.add<DeallocPattern>(typeConverter, ctx);

  // Control flow becomes pypto: scf.parallel/scf.forall map onto `pypto.spmd`
  // multi-core launches (the induction variable is the executing core's block
  // index) and scf.for onto the `pypto.range` per-core tile loop. Stage two
  // then only maps these pypto ops onto their emitpy spellings.
  patterns.add<ScfParallelPattern>(typeConverter, ctx);
  patterns.add<ScfForallPattern>(typeConverter, ctx);
  patterns.add<ScfForPattern>(typeConverter, ctx);
}

}  // namespace

void populateVectorToPyptoConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  populateFloatArithPatterns(patterns, typeConverter);
  populateIntArithPatterns(patterns, typeConverter);
  populateCmpSelectPatterns(patterns, typeConverter);
  populateCastPatterns(patterns, typeConverter);
  populateMathPatterns(patterns, typeConverter);
  populateTileAndControlFlowPatterns(patterns, typeConverter);

  // No func plumbing: signatures are rewritten by hand in the downstream
  // convert-pypto-to-emitpy pass (see ConvertVectorToPyptoPass::
  // runOnOperation for the body-level pre-pass), and returns/calls are
  // legal as-is.
}

}  // namespace vector_to_pypto

std::unique_ptr<Pass> createVectorToPyptoConversionPass() { return std::make_unique<ConvertVectorToPyptoPass>(); }

}  // namespace mlir
