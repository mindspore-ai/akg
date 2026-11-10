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

#include <algorithm>
#include <functional>
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

#include "mlir/Analysis/TopologicalSortUtils.h"
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

/// Function parameters have no per-operation padded shape. Operation patterns
/// compute their physical Tile types from the source operation and use the
/// conversion adaptor without a type-only mapping.
class VectorToPyptoTypeConverter : public TypeConverter {
 public:
  explicit VectorToPyptoTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](npuvector::NPUVectorType type) -> Type {
      return pypto::TileType::get(type.getContext(), type.getElementType(), type.getShape());
    });
  }
};

/// Preserve Tensor views and standard vectors; NPU tiles are represented by
/// the target dialect from the moment their defining operation is created.
static Type pyptoType(Type type) {
  if (auto vector = dyn_cast<npuvector::NPUVectorType>(type)) {
    return pypto::TileType::get(type.getContext(), vector.getElementType(), vector.getShape());
  }
  return type;
}

static Type pyptoType(Type type, ArrayRef<int64_t> shape) {
  if (isa<npuvector::NPUVectorType, pypto::TileType>(type)) {
    return pypto::TileType::get(type.getContext(), getElementTypeOrSelf(type), shape);
  }
  return type;
}

static Value mapped(ConversionPatternRewriter &rewriter, Value value) { return rewriter.getRemappedValue(value); }

static SmallVector<Value> mapped(ConversionPatternRewriter &rewriter, ValueRange values) {
  SmallVector<Value> result;
  std::transform(values.begin(), values.end(), std::back_inserter(result),
                 [&](Value value) { return mapped(rewriter, value); });
  return result;
}

static Type pyptoExtentType(Type source, ValueRange extents);
static Type pyptoLoadType(Type source, Value tensor, ValueRange extents);
static Type pyptoResultType(Value source);
static SmallVector<Type> pyptoResultTypes(Operation *source, ValueRange operands = {});

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
  // Tensor views retain their memref type; tile slices construct a target
  // Tile type while their original logical shape is still available.
  rewriter.replaceOpWithNewOp<pypto::SliceOp>(op, pyptoType(result.getType()), mapped(rewriter, source), window,
                                              mapped(rewriter, ValueRange(offsetValues)), dropDimsAttr);
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

/// One partial reduction a tile flows into: the axes that survive the
/// fold must carry whole 32-byte rows in the reduction's result dtype
/// (the output tile is laid out `[rows, 1]` col-major or `[1, cols]`
/// row-major, so a surviving axis becomes the contiguous one), so the
/// chain's allocation pads those axes by the result's element size
/// instead of the tile's own -- cross-dtype chains then agree on one
/// padded extent at their combining ops.
struct ReductionAxisPad {
  SmallVector<unsigned> axes;
  unsigned resultElemBytes = 0;
};

/// Forward walk from a tile value through the allocation-keeping
/// element-wise chain collecting every partial reduction it reaches;
/// see the definition below `keepsOperandAllocation`. Cycle-safe like
/// `collectSiblingAllocationBounds` below: merge edges between the
/// operands they combine form cycles a plain recursion never exits.
static void collectReductionAxisPadsImpl(Value v, unsigned depth, SmallVectorImpl<ReductionAxisPad> &pads,
                                          DenseSet<Value> &visited);
static void collectReductionAxisPads(Value v, unsigned depth, SmallVectorImpl<ReductionAxisPad> &pads);

/// The allocation extent for one tile axis: the contiguous (last) axis
/// pads to whole 32-byte rows in the tile's own dtype, and an axis that
/// survives a downstream partial reduction pads to whole rows in that
/// reduction's result dtype -- the larger of the two when both apply, so
/// cross-dtype chains sharing a reduction agree on one extent. Other
/// leading axes keep their raw extent.
static int64_t padAllocationExtent(int64_t extent, unsigned axis, unsigned rank, unsigned elemBytes,
                                   ArrayRef<ReductionAxisPad> pads) {
  int64_t aligned = extent;
  if (axis + 1 == rank && elemBytes) {
    aligned = alignExtentTo32B(aligned, elemBytes);
  }
  for (const auto &pad : pads) {
    if (llvm::is_contained(pad.axes, axis) && pad.resultElemBytes) {
      aligned = std::max(aligned, alignExtentTo32B(aligned, pad.resultElemBytes));
    }
  }
  return aligned;
}

/// Statically-known bounds (`maxSizes`/`paddedShapes`) with the
/// contiguous (last) axis and any reduction-surviving axis padded to
/// whole 32-byte rows the way the allocation will be; fails when a
/// bound is dynamic or the arity does not match the type's rank.
static std::optional<SmallVector<int64_t>> alignAllocationBounds(Value result, Type type, ValueRange bounds) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || bounds.size() != static_cast<size_t>(shaped.getRank())) {
    return std::nullopt;
  }
  unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
  SmallVector<ReductionAxisPad> pads;
  collectReductionAxisPads(result, 0, pads);
  SmallVector<int64_t> aligned;
  aligned.reserve(bounds.size());
  for (auto [index, bound] : llvm::enumerate(bounds)) {
    auto value = constIndexValue(bound);
    if (!value) {
      return std::nullopt;
    }
    // Only the contiguous axis and the reduction-surviving axes carry
    // whole-32-byte-row padding, mirroring what the allocation sites
    // do; other leading axes keep their raw bounds so cross-dtype
    // tiles with equal extents stay shape-compatible.
    aligned.push_back(padAllocationExtent(*value, index, bounds.size(), elemBytes, pads));
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
  if (isa<pypto::AddOp, pypto::SubOp, pypto::MulOp, pypto::DivOp, pypto::MaximumOp, pypto::MinimumOp,
          pypto::MaximumsOp, pypto::MinimumsOp, pypto::CastOp, pypto::AbsOp, pypto::ExpOp, pypto::LogOp,
          pypto::SqrtOp, pypto::RsqrtOp, pypto::RecipOp, pypto::NegOp, pypto::SinOp, pypto::CosOp, arith::AddIOp,
          arith::AddFOp, arith::MulIOp, arith::MulFOp, arith::SubIOp, arith::SubFOp, arith::DivSIOp, arith::DivUIOp,
          arith::DivFOp, arith::MaxSIOp, arith::MinSIOp, arith::ExtFOp, arith::TruncFOp, arith::NegFOp, arith::ExtSIOp,
          arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp,
          npuvector::ExtFOp, npuvector::TruncFOp, npuvector::ExtSIOp, npuvector::ExtUIOp, npuvector::TruncIOp,
          npuvector::SIToFPOp, npuvector::UIToFPOp, npuvector::FPToSIOp, npuvector::FPToUIOp, math::AbsFOp, math::ExpOp,
          math::LogOp, math::SqrtOp, math::RsqrtOp, math::SinOp, math::CosOp, math::RoundOp, math::FloorOp,
          math::CeilOp>(def)) {
    return true;
  }
  // Selects and comparisons keep the operands' allocation too: a dynamic
  // tile fed through a where/select or a comparison before a dynamic
  // reduction or store still names the read's bound.
  return isa<pypto::SelectOp, pypto::CmpOp, npuvector::SelectOp, arith::SelectOp, npuvector::CmpIOp, npuvector::CmpFOp,
             arith::MaximumFOp, arith::MinimumFOp>(def);
}

/// The result element size of a cast-family op (nullopt when `op` is
/// not one): an element-wise cast keeps the operand's allocation while
/// re-typing it, so the allocation must span whole 32-byte rows in the
/// cast's result dtype too -- an extent padded for the tile's own
/// 8-byte elements (140 rows = 1120B, on the grid) misses the grid once
/// the chain casts down to 4-byte elements (560B). Both the unconverted
/// arith / npuvector and the converted pypto spellings appear because
/// patterns run in arbitrary order.
static std::optional<unsigned> castResultElemBytes(Operation *op) {
  if (!isa<arith::ExtFOp, arith::TruncFOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::SIToFPOp,
           arith::UIToFPOp, arith::FPToSIOp, arith::FPToUIOp, npuvector::ExtFOp, npuvector::TruncFOp,
           npuvector::ExtSIOp, npuvector::ExtUIOp, npuvector::TruncIOp, npuvector::SIToFPOp, npuvector::UIToFPOp,
           npuvector::FPToSIOp, npuvector::FPToUIOp, pypto::CastOp>(op)) {
    return std::nullopt;
  }
  auto shaped = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!shaped) {
    return std::nullopt;
  }
  unsigned bytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
  return bytes ? std::optional<unsigned>(bytes) : std::nullopt;
}

/// A broadcast preserves row-alignment requirements when its source's
/// contiguous axis remains the result's contiguous axis without expanding
/// that logical dimension. Other broadcast axes do not constrain the source.
static bool broadcastPreservesContiguousAxis(Value source, Operation *user) {
  if (!isa<npuvector::BroadcastOp, vector::BroadcastOp>(user)) {
    return false;
  }
  auto sourceType = dyn_cast<ShapedType>(source.getType());
  auto resultType = dyn_cast<ShapedType>(user->getResult(0).getType());
  if (!sourceType || !resultType || sourceType.getRank() == 0 || resultType.getRank() == 0) {
    return false;
  }
  int64_t sourceLast = sourceType.getShape().back();
  int64_t resultLast = resultType.getShape().back();
  // Two dynamic dimensions do not prove equal runtime valid extents.
  if (ShapedType::isDynamic(sourceLast) || ShapedType::isDynamic(resultLast) || sourceLast != resultLast) {
    return false;
  }
  if (auto broadcast = dyn_cast<npuvector::BroadcastOp>(user)) {
    ArrayRef<int64_t> dims = broadcast.getDimension();
    return dims.size() == sourceType.getRank() && dims.back() == static_cast<int64_t>(resultType.getRank() - 1);
  }
  // vector.broadcast aligns its source axes with the trailing result axes.
  return true;
}

/// Forward walk from a tile value collecting the narrowest result dtype
/// width a cast chain downstream of it produces, or nullopt when no cast
/// consumes the chain. A cast keeps the source tile's allocation while
/// re-typing it (the frontend's `DeduceTileCastType`), so an allocation
/// padded for one dtype must stay 32-byte aligned at every narrower dtype
/// the chain later narrows to: a full allocated as INT32 over [2704] lays
/// aligned rows, yet the FP16-then-INT8 casts consuming it inherit
/// 2704-byte INT8 rows, which `pto.alloc_tile` rejects. The walk falls
/// through casts the way the reduction-axis one does, over the same
/// `keepsOperandAllocation` chain; depth-limited and cycle-guarded the
/// same way too.
static void collectCastChainMinElemBytesImpl(Value v, unsigned depth, std::optional<unsigned> &minBytes,
                                             DenseSet<Value> &visited) {
  if (depth > 256) {
    return;
  }
  if (!visited.insert(v).second) {
    return;
  }
  for (Operation *user : v.getUsers()) {
    if (auto bytes = castResultElemBytes(user)) {
      if (!minBytes || *bytes < *minBytes) {
        minBytes = bytes;
      }
    }
    if (keepsOperandAllocation(user) || broadcastPreservesContiguousAxis(v, user)) {
      for (Value result : user->getResults()) {
        collectCastChainMinElemBytesImpl(result, depth + 1, minBytes, visited);
      }
    }
  }
}

static std::optional<unsigned> castChainMinElemBytes(Value v) {
  std::optional<unsigned> minBytes;
  DenseSet<Value> visited;
  collectCastChainMinElemBytesImpl(v, 0, minBytes, visited);
  return minBytes;
}

/// Map a cast's row-alignment requirement across a broadcast's preserved
/// contiguous axis. In particular, a comparison broadcast before an INT8
/// cast must pad the compared data before constructing the packed mask.
static std::optional<ReductionAxisPad> broadcastCastAxisPad(Value source, Operation *user) {
  if (!broadcastPreservesContiguousAxis(source, user)) {
    return std::nullopt;
  }
  auto minBytes = castChainMinElemBytes(user->getResult(0));
  if (!minBytes) {
    return std::nullopt;
  }
  ReductionAxisPad pad;
  pad.axes.push_back(cast<ShapedType>(source.getType()).getRank() - 1);
  pad.resultElemBytes = *minBytes;
  return pad;
}

/// Collects the pad an unconverted partial reduction's output tile
/// allocation demands. `reduction_dims` names the one folded axis (the
/// first or the last, the restriction `rewritePartialReduction` imposes);
/// a full reduction yields a scalar, so no output tile allocation exists
/// to align.
static void collectNpuReductionPad(npuvector::ReductionOp reduction, SmallVectorImpl<ReductionAxisPad> &pads) {
  auto dims = reduction.getReductionDims();
  auto inputType = dyn_cast<ShapedType>(reduction.getVector().getType());
  auto resultType = dyn_cast<ShapedType>(reduction->getResult(0).getType());
  if (!dims || dims->size() != 1 || !inputType || inputType.getRank() < 2 || !resultType) {
    return;
  }
  ReductionAxisPad pad;
  unsigned rank = inputType.getRank();
  int64_t dim = (*dims)[0];
  if (dim == static_cast<int64_t>(rank - 1)) {
    // row: fold the last axis, every leading axis survives.
    for (unsigned axis = 0; axis + 1 < rank; ++axis) {
      pad.axes.push_back(axis);
    }
  } else if (dim == 0) {
    // col: fold the first axis, every trailing axis survives.
    for (unsigned axis = 1; axis < rank; ++axis) {
      pad.axes.push_back(axis);
    }
  } else {
    return;
  }
  pad.resultElemBytes = resultType.getElementType().getIntOrFloatBitWidth() / 8;
  if (pad.resultElemBytes) {
    pads.push_back(std::move(pad));
  }
}

/// Collects the pad a converted partial reduction's output tile allocation
/// demands: `axis` already spells row (fold the last axis) or col (fold the
/// first axis).
static void collectPyptoReductionPad(pypto::ReductionOp reduction, SmallVectorImpl<ReductionAxisPad> &pads) {
  auto inputType = dyn_cast<ShapedType>(reduction.getVector().getType());
  auto resultType = dyn_cast<ShapedType>(reduction->getResult(0).getType());
  if (!inputType || inputType.getRank() < 2 || !resultType) {
    return;
  }
  ReductionAxisPad pad;
  unsigned rank = inputType.getRank();
  if (reduction.getAxis() == pypto::ReduceAxis::row) {
    for (unsigned axis = 0; axis + 1 < rank; ++axis) {
      pad.axes.push_back(axis);
    }
  } else {
    for (unsigned axis = 1; axis < rank; ++axis) {
      pad.axes.push_back(axis);
    }
  }
  pad.resultElemBytes = resultType.getElementType().getIntOrFloatBitWidth() / 8;
  if (pad.resultElemBytes) {
    pads.push_back(std::move(pad));
  }
}

/// Forward walk from a tile value collecting every partial reduction it
/// flows into, through the element-wise/cast/select chain whose
/// allocation the tile keeps (`keepsOperandAllocation`). Both the
/// unconverted and the pypto reduction forms appear because patterns run
/// in arbitrary order. Any other consumer breaks the allocation chain --
/// the same ops `tileAllocationShape` refuses -- so the walk stops there
/// and the value stays unpadded; depth-limited like `tileAllocationShape`.
/// The value may reach a reduction as its vector operand or (on the
/// unconverted form) its accumulator: both combine into the reduction's
/// output tile, so both must carry the padded surviving axes.
static void collectReductionAxisPadsImpl(Value v, unsigned depth, SmallVectorImpl<ReductionAxisPad> &pads,
                                          DenseSet<Value> &visited) {
  // The budget must span the fused element-wise chains a scheduler emits:
  // a 50-input fused sum is ~55 keepsOperandAllocation hops from its head
  // reads to the loop-carried partial reduction, and any budget that cuts
  // the walk short leaves the mid-chain reads unpadded, mismatching the
  // padded tail at compile time. The chain stays linear in practice (one
  // user per link), so the headroom costs little.
  if (depth > 256 || !isa<ShapedType>(v.getType())) {
    return;
  }
  // Merge edges form cycles between the operands they combine (each
  // operand's walk reaches the other) and diamonds through shared
  // results, so a plain recursion revisits values forever. The pads
  // merge is a per-axis max, so one visit per value is enough.
  if (!visited.insert(v).second) {
    return;
  }
  for (Operation *user : v.getUsers()) {
    if (auto reduction = dyn_cast<npuvector::ReductionOp>(user)) {
      collectNpuReductionPad(reduction, pads);
      continue;
    }
    if (auto reduction = dyn_cast<pypto::ReductionOp>(user)) {
      collectPyptoReductionPad(reduction, pads);
      continue;
    }
    // An element-wise cast re-types the tile while keeping its
    // allocation, so the allocation must also satisfy the 32-byte grid
    // in the cast's result dtype: the contiguous axis carries that
    // requirement the same way it carries the tile's own row padding
    // and a reduction survivor's -- cross-dtype chains then agree on
    // one extent (144 rows is 1152B in int64 and 576B in fp32, whole
    // 32-byte rows on both grids; 140 is 1120B and 560B, the latter
    // off the fp32 grid). The walk falls through: a cast keeps the
    // allocation, so deeper casts and reductions still collect.
    if (auto castBytes = castResultElemBytes(user)) {
      auto shaped = dyn_cast<ShapedType>(v.getType());
      if (shaped && shaped.getRank() > 0) {
        ReductionAxisPad pad;
        pad.axes.push_back(shaped.getRank() - 1);
        pad.resultElemBytes = *castBytes;
        pads.push_back(std::move(pad));
      }
    }
    if (auto pad = broadcastCastAxisPad(v, user)) {
      pads.push_back(std::move(*pad));
    }
    if (keepsOperandAllocation(user)) {
      for (Value result : user->getResults()) {
        collectReductionAxisPadsImpl(result, depth + 1, pads, visited);
      }
    }
  }
}

static void collectReductionAxisPads(Value v, unsigned depth, SmallVectorImpl<ReductionAxisPad> &pads) {
  DenseSet<Value> visited;
  collectReductionAxisPadsImpl(v, depth, pads, visited);
}

/// Whether a user keeps the sibling-bounds walk alive: element-wise
/// allocation keepers plus the expand family, whose result keeps the tile
/// operand's allocation; partial reductions, whose vector and accumulator
/// operands combine into one output tile (so those operands are siblings
/// of each other -- a workspace accumulator read must cover the padded
/// input tile it merges with, an fp32 [6280] accumulator against a bf16
/// [6288] row -- and the result keeps the merged allocation; both the
/// unconverted and the pypto spellings appear because patterns run in
/// arbitrary order); and slices/merges, which keep the walk alive the
/// same way: an extract_slice result windows the source's allocation, so
/// a downstream merge it reaches still names the extent the source tile
/// must carry (a workspace read sliced onto the elementwise max with a
/// padded bf16 row inherits that row's bound), and an insert_slice
/// writes one shaped operand into the other's allocation, so the two
/// tiles must agree on one padded extent. Only the walk traverses them:
/// `tileAllocationShape` still refuses slices because a slice result
/// genuinely carries the smaller window, not the source's full
/// allocation.
static bool keepsSiblingWalkAlive(Operation *user) {
  return keepsOperandAllocation(user) ||
         isa<pypto::RowExpandAddOp, pypto::RowExpandSubOp, pypto::RowExpandMulOp, pypto::RowExpandDivOp,
             pypto::ColExpandAddOp, pypto::ColExpandSubOp, pypto::ColExpandMulOp, pypto::ColExpandDivOp>(user) ||
         isa<npuvector::ReductionOp, pypto::ReductionOp>(user) ||
         isa<npuvector::ExtractSliceOp, npuvector::InsertSliceOp, pypto::SliceOp>(user);
}

/// Forward walk from a tile value through the allocation-keeping chain
/// (`keepsOperandAllocation` plus the expand family, whose result keeps
/// the tile operand's allocation), aggregating per axis the allocation
/// of every sibling operand the chain merges with: PTOAS lays every tile
/// in the block onto one grid, so a read whose own chain carries no
/// reduction pad must still cover the allocation of the tiles it
/// combines with downstream -- e.g. the [8,*] a reduction-padded
/// sibling carries -- or the combine mismatches at compile time
/// ([7,*] vs [8,*]). Mirrors the sibling query `buildBroadcastExtents`
/// makes on a broadcast's direct consumers, but penetrates the chain
/// because a read's merge happens hops downstream. `siblingBounds`
/// stays empty until a static sibling bound appears; dynamic slots keep
/// the sentinel and never lower a bound.
static void collectSiblingAllocationBoundsImpl(Value v, unsigned depth, SmallVectorImpl<int64_t> &siblingBounds,
                                               DenseSet<Value> &visited) {
  auto shaped = dyn_cast<ShapedType>(v.getType());
  // The same long-chain headroom `collectReductionAxisPads` needs: this
  // cascade is what carries a reduction pad back up a fused chain past
  // the distance a direct reduction query covers, and a walk that stops
  // mid-chain leaves the head reads at the unpadded bound while the tail
  // already carries the padded one.
  if (depth > 256 || !shaped) {
    return;
  }
  // Merge edges form cycles between the operands they combine (each
  // operand's walk reaches the other), and diamonds through shared
  // results, so a plain recursion revisits values forever. The bounds
  // merge is a per-axis max, so one visit per value is enough.
  if (!visited.insert(v).second) {
    return;
  }
  unsigned rank = shaped.getRank();
  for (Operation *user : v.getUsers()) {
    if (!keepsSiblingWalkAlive(user)) {
      continue;
    }
    for (Value operand : user->getOperands()) {
      if (operand == v || !isa<ShapedType>(operand.getType())) {
        continue;
      }
      auto shape = tileAllocationShape(operand, 1);
      if (shape && shape->size() == rank) {
        if (siblingBounds.empty()) {
          siblingBounds.assign(rank, ShapedType::kDynamic);
        }
        for (unsigned axis = 0; axis < rank; ++axis) {
          if (ShapedType::isDynamic((*shape)[axis])) {
            continue;
          }
          siblingBounds[axis] =
            ShapedType::isDynamic(siblingBounds[axis]) ? (*shape)[axis] : std::max(siblingBounds[axis], (*shape)[axis]);
        }
      }
      // The backward query above only resolves the sibling's own
      // intrinsic boundary (a still-unconverted read pads from its
      // element size alone: fp32 104x4B already fills whole rows), but
      // that sibling's allocation is itself the merge of *its* further
      // siblings -- the [112] a head read carries comes from the
      // reduction result it merges with on another edge, which a plain
      // backward query never sees. Walk the sibling transitively so
      // the merge chain (read --mulf-- read --mulf-- reduction result)
      // propagates that bound all the way to the head.
      if (cast<ShapedType>(operand.getType()).getRank() == rank) {
        collectSiblingAllocationBoundsImpl(operand, depth + 1, siblingBounds, visited);
      }
    }
    for (Value result : user->getResults()) {
      collectSiblingAllocationBoundsImpl(result, depth + 1, siblingBounds, visited);
    }
  }
}

static void collectSiblingAllocationBounds(Value v, unsigned depth, SmallVectorImpl<int64_t> &siblingBounds) {
  DenseSet<Value> visited;
  collectSiblingAllocationBoundsImpl(v, depth, siblingBounds, visited);
}

/// The reads a backward allocation query is currently sibling-merging.
/// A read's sibling walk queries its merge siblings backward, and those
/// queries can land on reads whose own sibling walks reach back -- two
/// reads merged at one elementwise op reach each other through both
/// directions -- so an unguarded sibling merge recurses forever, and
/// letting every read the walk touches start its own walk multiplies the
/// query count per level. While any merge is in flight the queries it
/// issues resolve their intrinsic bounds only; the walk still collects
/// every other edge (transitively through the merge chain) it needs.
static DenseSet<Value> &activeSiblingQueryReads() {
  static thread_local DenseSet<Value> active;
  return active;
}

/// Merges the sibling bounds a read's own conversion pads it with into a
/// backward query that lands on it. `rewriteDynamicRead`/`rewriteDynamicLoad`
/// allocate the tile at its intrinsic 32-byte-aligned bound merged with the
/// allocation of every sibling its chain combines with downstream, so a
/// backward query that resolves only the intrinsic boundary (a fp32 read
/// whose 104x4B row already fills whole rows) under-reports what the final
/// program allocates -- the padded [112] a merge consumer or a lifted
/// broadcast reshape must spell.
static std::optional<SmallVector<int64_t>> mergeSiblingAllocationBounds(Value v, unsigned depth,
                                                                        std::optional<SmallVector<int64_t>> bounds) {
  if (!bounds) {
    return bounds;
  }
  DenseSet<Value> &active = activeSiblingQueryReads();
  if (!active.empty()) {
    return bounds;
  }
  active.insert(v);
  SmallVector<int64_t> siblingBounds;
  collectSiblingAllocationBounds(v, depth, siblingBounds);
  active.erase(v);
  if (siblingBounds.empty() || siblingBounds.size() != bounds->size()) {
    return bounds;
  }
  for (unsigned axis = 0; axis < bounds->size(); ++axis) {
    if (ShapedType::isDynamic(siblingBounds[axis]) || siblingBounds[axis] <= (*bounds)[axis]) {
      continue;
    }
    (*bounds)[axis] = siblingBounds[axis];
  }
  return bounds;
}

/// The per-axis largest allocation shape across the resolvable shaped
/// operands, or the first resolvable one when ranks disagree. Merging
/// rather than returning the first hit matters at element-wise merge
/// points: PTOAS runs the whole block on one grid, so the result's
/// allocation must cover every operand tile (e.g. an already padded bf16
/// load combined with an f32 load that pads later in conversion order).
static std::optional<SmallVector<int64_t>> allocationShapeFromOperands(Operation *def, unsigned depth) {
  std::optional<SmallVector<int64_t>> merged;
  // A select's condition never feeds the result's data: the frontend
  // (DeduceTileSelectType) takes the result's shape and valid extents
  // from the value operands only, so the mask's padded allocation must
  // not enlarge the merged result.
  bool skipCond = isa<pypto::SelectOp, npuvector::SelectOp, arith::SelectOp>(def);
  for (auto [index, operand] : llvm::enumerate(def->getOperands())) {
    if (skipCond && index == 0) {
      continue;
    }
    if (!isa<ShapedType>(operand.getType())) {
      continue;
    }
    auto bounds = tileAllocationShape(operand, depth + 1);
    if (!bounds) {
      continue;
    }
    if (!merged) {
      merged = bounds;
      continue;
    }
    if (merged->size() != bounds->size()) {
      continue;
    }
    for (unsigned axis = 0; axis < merged->size(); ++axis) {
      if (ShapedType::isDynamic((*bounds)[axis])) {
        continue;
      }
      (*merged)[axis] =
        ShapedType::isDynamic((*merged)[axis]) ? (*bounds)[axis] : std::max((*merged)[axis], (*bounds)[axis]);
    }
  }
  return merged;
}

/// The allocation a `pypto.load` names: the padded bounds it spells,
/// projected back onto the tile's rank when the load is rank-reduced.
static std::optional<SmallVector<int64_t>> loadAllocationShape(Value v, pypto::LoadOp load, unsigned depth) {
  // A rank-reduced load spells its padded bounds in the source's
  // coordinate system -- one entry per source dimension, every
  // dropped dimension spanning a single element padded and valid
  // alike (`rewritePaddedStaticRead`) -- so a query typed against
  // the tile's own rank projects the bounds back before aligning:
  // the mapped dimensions carry the tile's valid extents in tile
  // order, and the dropped unit dimensions fall out. Without the
  // projection the arity check inside `alignAllocationBounds`
  // rejects the bounds and the query degrades to the caller's
  // valid-extent fallback (8 instead of the padded 16 a bf16 row
  // allocates), misspelling the reshape a broadcast lift builds.
  ValueRange bounds = load.getPaddedShapes();
  auto shaped = dyn_cast<ShapedType>(v.getType());
  if (shaped && shaped.hasStaticShape() && bounds.size() > static_cast<size_t>(shaped.getRank()) &&
      load.getValidExtents().size() == bounds.size()) {
    SmallVector<Value> projected;
    size_t axis = 0;
    bool matched = true;
    for (auto [padded, valid] : llvm::zip(bounds, load.getValidExtents())) {
      if (axis < static_cast<size_t>(shaped.getRank())) {
        auto extent = constIndexValue(valid);
        if (extent && *extent == shaped.getDimSize(axis)) {
          projected.push_back(padded);
          ++axis;
          continue;
        }
      }
      auto unit = constIndexValue(padded);
      if (unit && *unit == 1) {
        continue;
      }
      matched = false;
      break;
    }
    if (matched && axis == static_cast<size_t>(shaped.getRank())) {
      return mergeSiblingAllocationBounds(v, depth, alignAllocationBounds(v, v.getType(), ValueRange(projected)));
    }
  }
  return mergeSiblingAllocationBounds(v, depth, alignAllocationBounds(v, v.getType(), bounds));
}

/// The allocation an `npuvector.broadcast` names, resolving a specialized
/// lift broadcast's identity-mapped axis through the source allocation.
static std::optional<SmallVector<int64_t>> broadcastAllocationShape(Value v, npuvector::BroadcastOp broadcast,
                                                                   unsigned depth) {
  // A specialized lift broadcast (rank-1 source lifted along one axis
  // onto a rank-2 result) materializes its identity-mapped axis from
  // the source allocation: the conversion reshapes the source chain
  // into [rows, 1]/[1, cols] with the extent `tileAllocationShape`
  // resolves on the source, so that axis carries the source's padded
  // allocation (140 where a 139-element load pads to whole 32-byte
  // rows) while the raw `maxSizes` bound stays unpadded. Resolve the
  // identity axis through the source when it has a static
  // allocation, so a query landing on an unconverted lift -- e.g. a
  // sibling constant broadcast materializing before the lift itself
  // converts -- names the allocation the expand family produces
  // instead of the smaller bound. Other axes keep the `maxSizes`
  // spelling padded the usual way.
  ArrayRef<int64_t> dimension = broadcast.getDimension();
  auto shaped = dyn_cast<ShapedType>(v.getType());
  auto srcType = dyn_cast<ShapedType>(broadcast.getSource().getType());
  // A scalar-source broadcast splats a rank-0 value across the result
  // and materializes no buffer of its own: the conversion folds it
  // back into a scalar operand of the consuming elementwise op, so
  // its `maxSizes` spell valid bounds rather than an allocation.
  // Resolving here would let `allocationShapeFromOperands` stop at
  // this operand and never reach the sibling that actually allocates
  // (an extf -> rank-reduced-load chain carrying padded bounds).
  // Decline so the query falls through to a real allocation source.
  if (!srcType || srcType.getRank() == 0) {
    return std::nullopt;
  }
  ValueRange maxSizes = broadcast.getMaxSizes();
  if (shaped && shaped.getRank() == 2 && srcType && srcType.getRank() == 1 && dimension.size() == 1 &&
      (dimension[0] == 0 || dimension[0] == 1) && maxSizes.size() == 2) {
    if (auto srcAlloc = tileAllocationShape(broadcast.getSource(), depth + 1); srcAlloc && srcAlloc->size() == 1) {
      unsigned identity = dimension[0] == 0 ? 0 : 1;
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      SmallVector<ReductionAxisPad> pads;
      collectReductionAxisPads(v, 0, pads);
      SmallVector<int64_t> aligned;
      aligned.reserve(maxSizes.size());
      for (auto [index, bound] : llvm::enumerate(maxSizes)) {
        auto value = constIndexValue(bound);
        if (!value) {
          return std::nullopt;
        }
        if (index == identity) {
          value = (*srcAlloc)[0];
        }
        aligned.push_back(padAllocationExtent(*value, index, maxSizes.size(), elemBytes, pads));
      }
      return aligned;
    }
  }
  return alignAllocationBounds(v, v.getType(), broadcast.getMaxSizes());
}

/// The allocation an extent-spelling op (`pypto.tile_full` /
/// `pypto.reshape`) names: a statically shaped result carries no extents
/// and names its own shape, while the per-dimension spelling covers every
/// axis.
static std::optional<SmallVector<int64_t>> extentListAllocationShape(Value v, ValueRange extents) {
  auto shaped = dyn_cast<ShapedType>(v.getType());
  if (shaped && shaped.hasStaticShape() && extents.empty()) {
    return SmallVector<int64_t>(shaped.getShape());
  }
  if (!shaped || extents.size() != static_cast<size_t>(shaped.getRank())) {
    return std::nullopt;
  }
  SmallVector<int64_t> shape;
  for (Value extent : extents) {
    auto bound = constIndexValue(extent);
    if (!bound) {
      return std::nullopt;
    }
    shape.push_back(*bound);
  }
  return shape;
}

/// The allocation an unconverted partial reduction's result keeps on the
/// surviving axes, mirroring `allocationShapeOfReduction` for the form
/// `rewritePartialReduction` later produces.
static std::optional<SmallVector<int64_t>> npuReductionAllocationShape(Value v, npuvector::ReductionOp reduction,
                                                                      unsigned depth) {
  auto shaped = dyn_cast<ShapedType>(v.getType());
  auto dims = reduction.getReductionDims();
  auto inputShaped = dyn_cast<ShapedType>(reduction.getVector().getType());
  if (!shaped || shaped.getRank() == 0 || !dims || dims->size() != 1 || !inputShaped || inputShaped.getRank() < 2) {
    // Full reductions (scalar results), multi-axis and inner-axis
    // reductions never map onto a partial row/col reduction.
    return std::nullopt;
  }
  auto bounds = tileAllocationShape(reduction.getVector(), depth + 1);
  if (!bounds || bounds->size() != static_cast<size_t>(inputShaped.getRank())) {
    return std::nullopt;
  }
  int64_t dim = (*dims)[0];
  // `rewritePartialReduction` maps only the first/last axis onto
  // pl.row_* / pl.col_*: row folds the last axis, col the first.
  size_t folded;
  if (dim == static_cast<int64_t>(inputShaped.getRank()) - 1) {
    folded = bounds->size() - 1;
  } else if (dim == 0) {
    folded = 0;
  } else {
    return std::nullopt;
  }
  SmallVector<int64_t> kept(*bounds);
  if (shaped.getRank() == static_cast<int64_t>(bounds->size())) {
    kept[folded] = 1;
  } else if (shaped.getRank() + 1 == static_cast<int64_t>(bounds->size())) {
    kept.erase(kept.begin() + folded);
  } else {
    return std::nullopt;
  }
  return kept;
}

static std::optional<SmallVector<int64_t>> blockArgumentAllocationShape(BlockArgument arg, unsigned depth) {
  if (auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp()); loop && arg.getArgNumber() > 0) {
    return tileAllocationShape(loop.getInitArgs()[arg.getArgNumber() - 1], depth + 1);
  }
  return std::nullopt;
}

static std::optional<SmallVector<int64_t>> npuReadAllocationShape(Value value, npuvector::TransferReadOp read,
                                                                  unsigned depth) {
  auto bounds = alignAllocationBounds(value, value.getType(), read.getMaxSizes());
  auto shaped = cast<ShapedType>(value.getType());
  if (!bounds && shaped.hasStaticShape()) {
    unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
    SmallVector<ReductionAxisPad> pads;
    collectReductionAxisPads(value, 0, pads);
    SmallVector<int64_t> shape;
    for (auto [axis, extent] : llvm::enumerate(shaped.getShape())) {
      shape.push_back(padAllocationExtent(extent, axis, shaped.getRank(), elemBytes, pads));
    }
    bounds = std::move(shape);
  }
  return mergeSiblingAllocationBounds(value, depth, bounds);
}

/// The statically allocatable shape covering a tile: static dims stay and
/// dynamic (`?`) dims resolve to the producing read/broadcast's `maxSizes`
/// bound, padded to whole 32-byte rows the way the allocation will be.
/// Element-wise users, casts and loop-carried accumulators keep the
/// allocation of the read they combine.
static std::optional<SmallVector<int64_t>> tileAllocationShape(Value v, unsigned depth) {
  // The budget covers the long element-wise chains a sibling query walks:
  // a polynomial (sin/cos Taylor) between a broadcast's consumer and the
  // load/carrier that carries the padded allocation spans dozens of
  // keepsOperandAllocation hops, and cutting the walk short leaves the
  // broadcast spelling the unpadded bound. The recursion stays linear in
  // practice -- `allocationShapeFromOperands` returns the first operand
  // that resolves -- so the headroom costs little.
  if (depth > 64) {
    return std::nullopt;
  }
  if (auto tile = dyn_cast<pypto::TileType>(v.getType());
      tile && llvm::none_of(tile.getShape(), ShapedType::isDynamic)) {
    return SmallVector<int64_t>(tile.getShape());
  }
  // Loop-carried tiles: an scf.for result keeps the allocation of the
  // yielded value, and an iter_arg keeps the init's.
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    return blockArgumentAllocationShape(arg, depth);
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    return std::nullopt;
  }
  if (auto read = dyn_cast<npuvector::TransferReadOp>(def)) {
    return npuReadAllocationShape(v, read, depth);
  }
  if (auto transpose = dyn_cast<npuvector::TransposeOp>(def)) {
    if (auto bounds = tileAllocationShape(transpose.getVector(), depth + 1)) {
      SmallVector<int64_t> shape;
      auto permutation = transpose.getPermutation();
      std::transform(permutation.begin(), permutation.end(), std::back_inserter(shape),
                     [&](int64_t axis) { return (*bounds)[axis]; });
      return shape;
    }
  }
  if (auto load = dyn_cast<pypto::LoadOp>(def)) {
    return loadAllocationShape(v, load, depth);
  }
  if (auto broadcast = dyn_cast<npuvector::BroadcastOp>(def)) {
    return broadcastAllocationShape(v, broadcast, depth);
  }
  if (auto full = dyn_cast<pypto::TileFullOp>(def)) {
    // A materialized carrier or constant allocates exactly the extents
    // the broadcast that created it spelled -- the padded bound the
    // sibling-aware `buildBroadcastExtents` computed. The per-dimension
    // spelling covers every axis; a statically shaped result carries no
    // extents and names its own shape.
    return extentListAllocationShape(v, full.getExtents());
  }
  if (auto reshape = dyn_cast<pypto::ReshapeOp>(def)) {
    // `pl.reshape` conserves physical elements, so the view's own
    // extents name the allocation the source carries: the lifted
    // [rows, 1]/[1, cols] views spell the source's padded bound (140
    // when the load padded 139 to whole 32-byte rows).
    return extentListAllocationShape(v, reshape.getExtents());
  }
  if (isa<pypto::RowExpandAddOp, pypto::RowExpandSubOp, pypto::RowExpandMulOp, pypto::RowExpandDivOp,
          pypto::ColExpandAddOp, pypto::ColExpandSubOp, pypto::ColExpandMulOp, pypto::ColExpandDivOp>(def)) {
    // The runtime broadcasts the vector onto the tile, so an expand's
    // result keeps the tile operand's allocation.
    return tileAllocationShape(def->getOperand(0), depth + 1);
  }
  if (auto reduction = dyn_cast<pypto::ReductionOp>(def)) {
    return allocationShapeOfReduction(v, reduction, depth);
  }
  if (auto reduction = dyn_cast<npuvector::ReductionOp>(def)) {
    return npuReductionAllocationShape(v, reduction, depth);
  }
  if (auto loop = dyn_cast<scf::ForOp>(def)) {
    return allocationShapeOfLoopResult(v, loop, depth);
  }
  if (keepsOperandAllocation(def)) {
    return allocationShapeFromOperands(def, depth);
  }
  return std::nullopt;
}

/// Query the source dataflow while it still carries logical valid shapes.
/// The result type is constructed once, when its target operation is built.
static Type pyptoResultType(Value source) {
  Type type = source.getType();
  if (!isa<npuvector::NPUVectorType>(type)) {
    return type;
  }
  if (auto shape = tileAllocationShape(source)) {
    return pyptoType(type, *shape);
  }
  return pyptoType(type);
}

static SmallVector<Type> pyptoResultTypes(Operation *source, ValueRange operands) {
  SmallVector<Type> result;
  for (Value value : source->getResults()) {
    Type type = pyptoResultType(value);
    if (isa<npuvector::NPUVectorType>(value.getType())) {
      for (Value operand : operands) {
        if (auto tile = dyn_cast<pypto::TileType>(operand.getType())) {
          type = pyptoType(value.getType(), tile.getShape());
          break;
        }
      }
    }
    result.push_back(type);
  }
  return result;
}

static Type pyptoExtentType(Type source, ValueRange extents) {
  if (!isa<npuvector::NPUVectorType, pypto::TileType>(source) || extents.empty()) {
    return pyptoType(source);
  }
  SmallVector<int64_t> shape;
  std::transform(extents.begin(), extents.end(), std::back_inserter(shape),
                 [](Value extent) { return constIndexValue(extent).value_or(ShapedType::kDynamic); });
  return pyptoType(source, shape);
}

static Type pyptoLoadType(Type source, Value tensor, ValueRange extents) {
  Type result = pyptoExtentType(source, extents);
  if (extents.empty()) {
    if (auto tile = dyn_cast<pypto::TileType>(result)) {
      if (auto input = dyn_cast<ShapedType>(tensor.getType()); input && input.getRank() > tile.getRank()) {
        SmallVector<int64_t> shape(tile.getShape());
        shape.insert(shape.begin(), input.getRank() - shape.size(), 1);
        return pyptoType(result, shape);
      }
    }
  }
  return result;
}

/// Views `input` as a `[rows, cols]` tile (`pypto.reshape`);
/// convert-pypto-to-emitpy lowers it into `pl.reshape(input, [rows, cols])`.
static Value reshapeTileOp(ConversionPatternRewriter &rewriter, Location loc, Value input, int64_t rows, int64_t cols,
                           Type elemType) {
  assert(pyptoDtypeName(elemType) && "caller validated the element type");
  return rewriter.create<pypto::ReshapeOp>(
    loc,
    pyptoExtentType(pypto::TileType::get(rewriter.getContext(), elemType, SmallVector<int64_t>{rows, cols}),
                    ValueRange()),
    mapped(rewriter, input), ValueRange(), ValueRange());
}

/// Views the physical allocation as `[rows, cols]` while carrying the
/// smaller valid region separately. The Tile type preserves the physical
/// element count; valid extents lower to a slice that bounds the store.
static Value reshapeTileWithValidOp(ConversionPatternRewriter &rewriter, Location loc, Value input, int64_t rows,
                                    int64_t cols, int64_t validRows, int64_t validCols, Type elemType) {
  assert(pyptoDtypeName(elemType) && "caller validated the element type");
  SmallVector<Value> extents{rewriter.create<arith::ConstantIndexOp>(loc, rows),
                              rewriter.create<arith::ConstantIndexOp>(loc, cols)};
  SmallVector<Value> validExtents{rewriter.create<arith::ConstantIndexOp>(loc, validRows),
                                  rewriter.create<arith::ConstantIndexOp>(loc, validCols)};
  return rewriter.create<pypto::ReshapeOp>(
    loc,
    pyptoExtentType(pypto::TileType::get(rewriter.getContext(), elemType, SmallVector<int64_t>{rows, cols}),
                    ValueRange(extents)),
    mapped(rewriter, input), mapped(rewriter, ValueRange(extents)), mapped(rewriter, ValueRange(validExtents)));
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

static bool fuseScalarBroadcastConsumer(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter);
static bool fuseTileBroadcastConsumer(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter);

/// Rewrites an operation into the matching pypto op, e.g.
/// `arith.addf %a, %b` -> `pypto.add %a, %b`.
/// The adaptor supplies the converted data tiles. Their physical shape and
/// the source result's element type determine the target type at creation.
/// Scalar operands keep the arith op: pypto is a tile dialect, and scalar
/// arithmetic rides through to stage two, which maps it onto infix
/// pseudo-callees named after the arith ops.
template <typename SourceOp, typename PyptoOp>
class OpToPyptoOpPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op->getOperands(), [](Value operand) {
          return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(operand.getType());
        })) {
      return rewriter.notifyMatchFailure(op, "scalar arithmetic stays arith");
    }
    if (fuseScalarBroadcastConsumer(op, adaptor.getOperands(), rewriter)) {
      return success();
    }
    if (fuseTileBroadcastConsumer(op, adaptor.getOperands(), rewriter)) {
      return success();
    }
    rewriter.replaceOpWithNewOp<PyptoOp>(op, pyptoResultTypes(op, adaptor.getOperands()), adaptor.getOperands());
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
    if (isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(op.getLhs().getType())) {
      rewriter.replaceOpWithNewOp<TileOp>(op, pyptoResultTypes(op, adaptor.getOperands()), adaptor.getOperands());
    } else {
      rewriter.replaceOpWithNewOp<ScalarOp>(op, pyptoResultTypes(op, adaptor.getOperands()), adaptor.getOperands());
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
    if (!isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(op.getLhs().getType())) {
      return rewriter.notifyMatchFailure(op, "scalar remainder waits for stage two");
    }
    rewriter.replaceOpWithNewOp<pypto::FmodOp>(op, pyptoResultTypes(op, adaptor.getOperands()), adaptor.getOperands());
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

/// Select conditions and broadcasts consume or relay the packed mask.
static bool keepsPackedMask(OpOperand &use) {
  // A broadcast relays the comparison mask to the select's geometry. Its
  // own rewrite unpacks any elementwise uses at that final geometry.
  if (isa<npuvector::BroadcastOp, vector::BroadcastOp>(use.getOwner())) {
    return true;
  }
  return use.getOperandNumber() == 0 && isa<arith::SelectOp, npuvector::SelectOp, pypto::SelectOp>(use.getOwner());
}

/// Query only the consumers that will read the elementwise 0/1 tile.
static std::optional<unsigned> unpackCastChainMinElemBytes(Value result) {
  std::optional<unsigned> minBytes;
  DenseSet<Value> visited;
  for (OpOperand &use : result.getUses()) {
    if (keepsPackedMask(use)) {
      continue;
    }
    Operation *user = use.getOwner();
    if (auto bytes = castResultElemBytes(user); bytes && (!minBytes || *bytes < *minBytes)) {
      minBytes = bytes;
    }
    if (keepsOperandAllocation(user)) {
      for (Value next : user->getResults()) {
        collectCastChainMinElemBytesImpl(next, 0, minBytes, visited);
      }
    }
  }
  return minBytes;
}

/// Legacy vector types name logical elements, so their physical allocation
/// remains explicit. PyPTO tile types already carry the final allocation.
static SmallVector<int64_t> maskUnpackAllocation(Operation *replaced, Value mask) {
  SmallVector<int64_t> allocation(cast<ShapedType>(mask.getType()).getShape());
  if (!isa<VectorType>(mask.getType())) {
    return allocation;
  }
  if (auto inferred = tileAllocationShape(mask)) {
    allocation = std::move(*inferred);
  }
  if (auto minBytes = unpackCastChainMinElemBytes(replaced->getResult(0)); minBytes && !allocation.empty()) {
    allocation.back() = alignExtentTo32B(allocation.back(), *minBytes);
  }
  return allocation;
}

/// A2/A3 select/expand has no 8-bit form, so unpack BOOL masks into INT32
/// branches. Casts consuming that integer tile derive their own dtype.
static Type unpackedMaskType(Type maskType, MLIRContext *ctx) {
  Type elem = IntegerType::get(ctx, 32);
  if (auto tile = dyn_cast<pypto::TileType>(maskType)) {
    return pypto::TileType::get(ctx, elem, tile.getShape());
  }
  if (auto nv = dyn_cast<npuvector::NPUVectorType>(maskType)) {
    return pyptoType(nv.cloneWith(nv.getShape(), elem));
  }
  return VectorType::get(cast<ShapedType>(maskType).getShape(), elem);
}

/// Construct each branch once at the mask's final physical allocation,
/// retaining the original logical valid region independently. Narrow-cast
/// row alignment is propagated to the producers before the mask is built;
/// padding branches afterwards would disagree with the mask and select.
static Value createMaskUnpackBranch(Location loc, Type type, ArrayRef<int64_t> allocation, ArrayRef<int64_t> validShape,
                                    StringRef value, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> extents;
  bool explicitAllocation = allocation != cast<ShapedType>(type).getShape();
  if (explicitAllocation || validShape != allocation) {
    std::transform(allocation.begin(), allocation.end(), std::back_inserter(extents),
                   [&](int64_t physical) { return rewriter.create<arith::ConstantIndexOp>(loc, physical); });
  }
  Value full = rewriter.create<pypto::TileFullOp>(loc, type, rewriter.getStringAttr(value),
                                                  explicitAllocation ? ValueRange(extents) : ValueRange{});
  if (validShape == allocation) {
    return full;
  }
  SmallVector<Value> valid;
  std::transform(validShape.begin(), validShape.end(), std::back_inserter(valid),
                 [&](int64_t actual) { return rewriter.create<arith::ConstantIndexOp>(loc, actual); });
  return rewriter.create<pypto::ReshapeOp>(loc, type, full, ValueRange(extents), ValueRange(valid));
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
  bool needsUnpack =
    llvm::any_of(replaced->getResult(0).getUses(), [&](OpOperand &use) { return !keepsPackedMask(use); });
  auto shaped = dyn_cast<ShapedType>(mask.getType());
  if (!needsUnpack || !shaped || !shaped.hasStaticShape()) {
    rewriter.replaceOp(replaced, mask);
    return;
  }
  Location loc = replaced->getLoc();
  Type unpackType = unpackedMaskType(mask.getType(), rewriter.getContext());
  ArrayRef<int64_t> validShape = shaped.getShape();
  if (auto logical = dyn_cast<npuvector::NPUVectorType>(replaced->getResult(0).getType())) {
    validShape = logical.getShape();
  }
  SmallVector<int64_t> allocation = maskUnpackAllocation(replaced, mask);
  Value ones = createMaskUnpackBranch(loc, unpackType, allocation, validShape, "1", rewriter);
  Value zeros = createMaskUnpackBranch(loc, unpackType, allocation, validShape, "0", rewriter);
  Value unpacked = rewriter.create<pypto::SelectOp>(loc, unpackType, mask, ones, zeros);
  // Rebind elementwise consumers separately: the comparison's own
  // replacement remains the same-typed packed mask for selects/broadcasts.
  SmallVector<OpOperand *> unpackUses;
  for (OpOperand &use : replaced->getResult(0).getUses()) {
    if (!keepsPackedMask(use)) {
      unpackUses.push_back(&use);
    }
  }
  rewriter.replaceOp(replaced, mask);
  for (OpOperand *use : unpackUses) {
    rewriter.modifyOpInPlace(use->getOwner(), [&]() { use->set(unpacked); });
  }
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
    if (llvm::none_of(op->getOperands(), [](Value operand) {
          return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(operand.getType());
        })) {
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
    if (fuseScalarBroadcastConsumer(op, adaptor.getOperands(), rewriter)) {
      return success();
    }
    auto predicate = static_cast<pypto::CmpPredicate>(*code);
    Location loc = op->getLoc();
    Type resultType = pyptoResultTypes(op, adaptor.getOperands()).front();
    Value mask = rewriter.create<pypto::CmpOp>(loc, pyptoType(resultType), predicate, adaptor.getOperands()[0],
                                               adaptor.getOperands()[1]);
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
    Type resultType = pyptoResultTypes(op, adaptor.getOperands()).front();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op, "cast target type has no PyPTO dtype");
    }
    if (op->getOperand(0).getType() == resultType) {
      // A comparison unpacking already materialized this dtype (the mask
      // select spells its branches in i32); ptoas rejects a same-dtype
      // tile cast outright, so the cast folds into a plain forwarding.
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::CastOp>(op, pyptoType(resultType), adaptor.getOperands()[0]);
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
    Type resultType = pyptoResultTypes(op, adaptor.getOperands()).front();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op, "cast target type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::ReinterpretViewOp>(op, pyptoType(resultType), adaptor.getOperands()[0]);
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
      return rewriteSplatConstant(op, dense, type, rewriter);
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
    rewriter.replaceOpWithNewOp<pypto::ConstantOp>(op, pyptoType(type), rewriter.getStringAttr(*text));
    return success();
  }

 private:
  /// Maps a splat tile constant onto `pl.tile.full(shape, dtype, value)`,
  /// raising the allocation to the padded bounds of the sibling tiles it
  /// merges with downstream.
  LogicalResult rewriteSplatConstant(arith::ConstantOp op, DenseElementsAttr dense, Type type,
                                     ConversionPatternRewriter &rewriter) const {
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
    // PTOAS lays every tile in the block onto one grid, so the constant
    // must allocate the padded shape of the siblings it merges with
    // downstream (e.g. a [3060] one against a Taylor chain over a load
    // that pads to [3064]) or the merge mismatches at compile time.
    // Spell the raised bound as `tile.full` extents; the result type
    // keeps naming the tile's own shape and an unraised constant keeps
    // the extents-free spelling.
    SmallVector<int64_t> siblingBounds;
    collectSiblingAllocationBounds(op.getResult(), 0, siblingBounds);
    ValueRange extents;
    SmallVector<Value> raised;
    bool needsPad = siblingBounds.size() == static_cast<size_t>(shaped.getRank());
    for (unsigned axis = 0; needsPad && axis < shaped.getRank(); ++axis) {
      needsPad = !ShapedType::isDynamic(siblingBounds[axis]) && siblingBounds[axis] > shaped.getDimSize(axis);
    }
    if (needsPad) {
      for (unsigned axis = 0; axis < shaped.getRank(); ++axis) {
        int64_t size = shaped.getDimSize(axis);
        if (!ShapedType::isDynamic(siblingBounds[axis])) {
          size = std::max(size, siblingBounds[axis]);
        }
        raised.push_back(rewriter.create<arith::ConstantIndexOp>(op.getLoc(), size));
      }
      extents = ValueRange(raised);
    }
    rewriter.replaceOpWithNewOp<pypto::TileFullOp>(op, pyptoExtentType(type, extents), rewriter.getStringAttr(*elem),
                                                   extents);
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
    bool plainRead = elemBytes == 0 || cols <= 0 || (cols * static_cast<int64_t>(elemBytes)) % 32 == 0;
    if (plainRead) {
      // PTOAS lays every tile in the block onto one grid, so even a
      // whole-32-byte-row read must still cover the allocation of the
      // tiles it merges with downstream: an fp32 [6280] workspace read
      // (6280*4 = 785 whole rows) combining with a bf16 [6288] sibling
      // (6280*2 bytes pad up to 393 rows, kept through the cast)
      // mismatches at compile time. A sibling bound above the read's own
      // extent reroutes through the padded/valid form so both sides
      // allocate one shape, while the valid extent keeps bounding the
      // data actually moved.
      SmallVector<int64_t> siblingBounds;
      collectSiblingAllocationBounds(op.getVector(), 0, siblingBounds);
      for (unsigned dim = 0; dim < shaped.getRank(); ++dim) {
        if (dim < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[dim]) &&
            siblingBounds[dim] > shaped.getDimSize(dim)) {
          plainRead = false;
          break;
        }
      }
    }
    if (plainRead) {
      return rewritePlainRead(op, shaped, rewriter);
    }
    return rewritePaddedStaticRead(op, shaped, rewriter);
  }

 private:
  /// Rewrites a statically shaped, plain (unpadded) read into `pl.load`,
  /// restoring the tile's own shape with a `pypto.reshape` when the read is
  /// rank-reduced.
  LogicalResult rewritePlainRead(TransferReadOp op, ShapedType shaped, ConversionPatternRewriter &rewriter) const {
    // A rank-reduced read (source rank > tile rank) keeps addressing
    // every source dimension: `pl.load` shares the source's coordinate
    // system, and the runtime then returns a tile with the source's
    // rank while downstream ops are typed against the tile's own
    // (physical) rank -- restore the tile's shape with a `pypto.reshape`
    // right after the load. Reads that drop leading source dimensions
    // keep the 3-argument form: convert-pypto-to-emitpy reconstructs
    // the source-coordinate shape by padding the tile's shape with
    // leading ones. Any other projection must spell the shape out --
    // the tile's extents at their projected source dimensions, 1 along
    // the dropped ones -- because the leading-ones guess would misplace
    // the extents (e.g. drop a middle dimension).
    auto srcType = dyn_cast<ShapedType>(op.getSource().getType());
    if (srcType && srcType.getRank() > shaped.getRank()) {
      unsigned srcRank = srcType.getRank();
      unsigned tileRank = shaped.getRank();
      SmallVector<unsigned> tileToSrc = tileToSourceDims(op, tileRank, srcRank);
      bool leadingDrop = tileToSrc.size() == tileRank;
      for (unsigned dim = 0; dim < tileRank && leadingDrop; ++dim) {
        leadingDrop = tileToSrc[dim] == srcRank - tileRank + dim;
      }
      if (tileToSrc.size() == tileRank && !leadingDrop) {
        Location loc = op.getLoc();
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        SmallVector<Value> shapeValues(srcRank, one);
        for (unsigned tileDim = 0; tileDim < tileRank; ++tileDim) {
          shapeValues[tileToSrc[tileDim]] =
            rewriter.create<arith::ConstantIndexOp>(loc, shaped.getDimSize(tileDim));
        }
        Value load = rewriter.create<pypto::LoadOp>(
          loc, pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange(shapeValues)),
          mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()),
          mapped(rewriter, ValueRange(shapeValues)), mapped(rewriter, ValueRange(shapeValues)));
        rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, pyptoExtentType(op.getVector().getType(), ValueRange()), load,
                                                      ValueRange(), ValueRange());
        return success();
      }
      Value load = rewriter.create<pypto::LoadOp>(
        op.getLoc(), pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange{}),
        mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()), ValueRange{}, ValueRange{});
      rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, pyptoExtentType(op.getVector().getType(), ValueRange()), load,
                                                    ValueRange(), ValueRange());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(
      op, pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange{}), mapped(rewriter, op.getSource()),
      mapped(rewriter, op.getIndices()), ValueRange{}, ValueRange{});
    return success();
  }

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
    SmallVector<ReductionAxisPad> pads;
    collectReductionAxisPads(op.getVector(), 0, pads);
    SmallVector<int64_t> siblingBounds;
    collectSiblingAllocationBounds(op.getVector(), 0, siblingBounds);
    for (unsigned tileDim = 0; tileDim < tileRank; ++tileDim) {
      int64_t size = shaped.getDimSize(tileDim);
      int64_t bound = padAllocationExtent(size, tileDim, tileRank, elemBytes, pads);
      // The sibling grid raises the bound the same way the broadcast
      // side's sibling query does.
      if (tileDim < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[tileDim])) {
        bound = std::max(bound, siblingBounds[tileDim]);
      }
      paddedValues[tileToSrc[tileDim]] = rewriter.create<arith::ConstantIndexOp>(loc, bound);
      validValues[tileToSrc[tileDim]] = rewriter.create<arith::ConstantIndexOp>(loc, size);
    }
    // Rank reduction: the load shares the source's coordinate system, so
    // the runtime hands back a tile of the source's rank while downstream
    // ops are typed against the tile's own rank -- restore it with a
    // reshape right after the load, as the other read paths do. The
    // reshape's extents are the tile's allocation bounds (the values at
    // the tile dims' source positions; the element counts match: every
    // dropped source dimension spans a single element).
    if (srcRank > tileRank) {
      SmallVector<Value> tileExtents;
      for (unsigned tileDim = 0; tileDim < tileRank; ++tileDim) {
        tileExtents.push_back(paddedValues[tileToSrc[tileDim]]);
      }
      Value load = rewriter.create<pypto::LoadOp>(
        op.getLoc(), pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange(paddedValues)),
        mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()), mapped(rewriter, ValueRange(paddedValues)),
        mapped(rewriter, ValueRange(validValues)));
      rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op,
                                                    pyptoExtentType(op.getVector().getType(), ValueRange(tileExtents)),
                                                    load, mapped(rewriter, ValueRange(tileExtents)), ValueRange());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(
      op, pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange(paddedValues)),
      mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()), mapped(rewriter, ValueRange(paddedValues)),
      mapped(rewriter, ValueRange(validValues)));
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
    unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
    SmallVector<ReductionAxisPad> pads;
    collectReductionAxisPads(op.getVector(), 0, pads);
    SmallVector<int64_t> siblingBounds;
    collectSiblingAllocationBounds(op.getVector(), 0, siblingBounds);
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
        alloc = padDynamicAllocation(alloc, dim, tileRank, elemBytes, pads, siblingBounds, rewriter, loc);
        shapeValues.push_back(alloc);
        validValues.push_back(extent);
        continue;
      }
      // A static axis surviving a downstream reduction pads too: the
      // allocation must cover the reduction's result rows, while the
      // valid extent keeps bounding the data actually moved.
      int64_t padded = padAllocationExtent(size, dim, tileRank, elemBytes, pads);
      if (dim < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[dim])) {
        padded = std::max(padded, siblingBounds[dim]);
      }
      Value extent = rewriter.create<arith::ConstantIndexOp>(loc, padded);
      Value valid = rewriter.create<arith::ConstantIndexOp>(loc, size);
      shapeValues.push_back(extent);
      validValues.push_back(valid);
    }
    // Rank reduction: dropped source dimensions span a single element;
    // the tile's axes carry their extents at the source dimensions the
    // permutation map names.
    if (srcRank > tileRank) {
      emitRankReducedDynamicLoad(op, shaped, rewriter, shapeValues, validValues, tileToSrc, srcRank);
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(
      op, pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange(shapeValues)),
      mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()), mapped(rewriter, ValueRange(shapeValues)),
      mapped(rewriter, ValueRange(validValues)));
    return success();
  }

  /// Pads a dynamic read axis's allocation bound to whole 32-byte rows and
  /// to the sibling grid, returning the padded value (a new constant only
  /// when padding actually changes the bound). PTOAS allocates whole
  /// 32-byte rows on the contiguous (last) dimension and on the axes
  /// surviving a downstream partial reduction (padded by that reduction's
  /// result dtype, so the cross-dtype tiles a reduction combines agree on
  /// one extent); the valid extent still bounds the data actually moved.
  /// Other leading dimensions carry no row-padding requirement, so a
  /// cross-dtype tile with equal extents stays shape-compatible with the
  /// tiles it combines with. The sibling grid raises the padded bound the
  /// same way the broadcast side's sibling query does, so tiles merging
  /// downstream share one allocation shape.
  Value padDynamicAllocation(Value alloc, unsigned dim, unsigned tileRank, unsigned elemBytes,
                             const SmallVector<ReductionAxisPad> &pads,
                             const SmallVector<int64_t> &siblingBounds, ConversionPatternRewriter &rewriter,
                             Location loc) const {
    if (auto bound = constIndexValue(alloc)) {
      int64_t padded = padAllocationExtent(*bound, dim, tileRank, elemBytes, pads);
      if (dim < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[dim])) {
        padded = std::max(padded, siblingBounds[dim]);
      }
      if (padded != *bound) {
        return rewriter.create<arith::ConstantIndexOp>(loc, padded);
      }
    }
    return alloc;
  }

  /// Emits the load for a rank-reduced dynamic read, restoring the tile's
  /// own rank with a reshape right after the load, as the statically
  /// shaped path does.
  void emitRankReducedDynamicLoad(npuvector::TransferReadOp op, ShapedType shaped,
                                  ConversionPatternRewriter &rewriter, const SmallVector<Value> &shapeValues,
                                  const SmallVector<Value> &validValues, ArrayRef<unsigned> tileToSrc,
                                  unsigned srcRank) const {
    Location loc = op.getLoc();
    // The load shares the source's coordinate system, so the runtime
    // hands back a tile of the source's rank while downstream ops are
    // typed against the tile's own rank -- restore it with a reshape
    // right after the load. The reshape's extents are the tile's
    // allocation bounds, the same values the load allocates (the
    // element counts match: every dropped source dimension spans a
    // single element).
    SmallVector<Value> tileExtents = shapeValues;
    SmallVector<Value> srcShape(srcRank);
    SmallVector<Value> srcValid(srcRank);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    for (unsigned dim = 0; dim < srcRank; ++dim) {
      srcShape[dim] = one;
      srcValid[dim] = one;
    }
    for (unsigned tileDim = 0; tileDim < shaped.getRank(); ++tileDim) {
      srcShape[tileToSrc[tileDim]] = shapeValues[tileDim];
      srcValid[tileToSrc[tileDim]] = validValues[tileDim];
    }
    Value load =
      rewriter.create<pypto::LoadOp>(loc, pyptoLoadType(op.getVector().getType(), op.getSource(), ValueRange(srcShape)),
                                     mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()),
                                     mapped(rewriter, ValueRange(srcShape)), mapped(rewriter, ValueRange(srcValid)));
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op,
                                                  pyptoExtentType(op.getVector().getType(), ValueRange(tileExtents)),
                                                  load, mapped(rewriter, ValueRange(tileExtents)), ValueRange());
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
    // never flattened and fails PTO's 2D valid-shape check. The tile's N
    // lands on the destination dim the permutation map projects, so
    // transposed stores place it on a leading dim. Higher-rank tiles are
    // already 2D-collapsible and keep the default store.
    if (valueType && destType && valueType.getRank() == 1 && destType.getRank() > 2) {
      if (failed(storeRankReducedPadded(rewriter, loc, op, vector, destType, valueType))) {
        return failure();
      }
      finishRewrite(rewriter, op);
      return success();
    }

    rewriter.create<pypto::StoreOp>(loc, mapped(rewriter, op.getVector()), mapped(rewriter, op.getSource()),
                                    mapped(rewriter, op.getIndices()));
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
    return rewriter
      .create<pypto::ReadOp>(loc, pyptoType(elemType), mapped(rewriter, tile), mapped(rewriter, ValueRange(coord)))
      .getResult();
  }

  /// Writes one scalar into `dest` at `indices` with `pl.write`.
  static void writeScalar(ConversionPatternRewriter &rewriter, Location loc, Value dest, ValueRange indices,
                          Value scalar) {
    SmallVector<Value> offsets(indices.begin(), indices.end());
    if (auto shaped = dyn_cast<ShapedType>(dest.getType())) {
      while (offsets.size() < static_cast<size_t>(shaped.getRank())) {
        offsets.insert(offsets.begin(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
    }
    rewriter.create<pypto::WriteOp>(loc, mapped(rewriter, dest), mapped(rewriter, scalar),
                                    mapped(rewriter, ValueRange(offsets)));
  }

  /// Writes a scalar into a single-element destination with `pl.write`,
  /// addressing every destination dimension at 0.
  static void writeScalarToSingleElement(ConversionPatternRewriter &rewriter, Location loc, Value dest, Value scalar,
                                         ShapedType destType) {
    SmallVector<Value> offsets;
    for (int64_t d = 0; d < destType.getRank(); ++d) {
      offsets.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
    rewriter.create<pypto::WriteOp>(loc, mapped(rewriter, dest), mapped(rewriter, scalar),
                                    mapped(rewriter, ValueRange(offsets)));
  }

  /// The allocation bound of a dynamic rank-1 tile produced by an
  /// unconverted `transfer_read`: its first `maxSizes` entry, padded the
  /// same way `rewriteDynamicRead` pads the read it rewrites -- whole
  /// 32-byte rows in the tile's own dtype, any reduction-surviving pad
  /// the chain carries, and the sibling grid its merges share -- so the
  /// bound reported here is the bound that rewrite allocates.
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
      SmallVector<ReductionAxisPad> pads;
      collectReductionAxisPads(read.getVector(), 0, pads);
      SmallVector<int64_t> siblingBounds;
      collectSiblingAllocationBounds(read.getVector(), 0, siblingBounds);
      int64_t padded = padAllocationExtent(*bound, 0, shaped.getRank(), elemBytes, pads);
      if (!siblingBounds.empty() && !ShapedType::isDynamic(siblingBounds[0])) {
        padded = std::max(padded, siblingBounds[0]);
      }
      return padded;
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
               pypto::MaximumsOp, pypto::MinimumsOp, pypto::AbsOp, pypto::ExpOp, pypto::LogOp, pypto::SqrtOp,
               pypto::RsqrtOp, pypto::RecipOp, pypto::NegOp, pypto::SinOp, pypto::CosOp>(op);
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
  static std::optional<int64_t> npuvectorReductionExtent(npuvector::ReductionOp reduction, Value v, unsigned depth) {
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
  /// their operand's allocation. A select's condition is skipped: the
  /// frontend derives the result's shape and valid extents from the value
  /// operands only, so the mask's allocation must not win here either.
  static std::optional<int64_t> operandTileExtent(Operation *owner, unsigned depth) {
    bool skipCond = isa<pypto::SelectOp, npuvector::SelectOp, arith::SelectOp>(owner);
    for (auto [index, operand] : llvm::enumerate(owner->getOperands())) {
      if (skipCond && index == 0) {
        continue;
      }
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
    if (isa<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp, arith::SubIOp, arith::SubFOp, arith::DivSIOp,
            arith::DivUIOp, arith::DivFOp, arith::MaxSIOp, arith::MinSIOp, math::AbsFOp, math::ExpOp, math::LogOp,
            math::SqrtOp, math::RsqrtOp, math::SinOp, math::CosOp, math::RoundOp, math::FloorOp, math::CeilOp>(op)) {
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
    if (isa<pypto::SelectOp, pypto::CmpOp, npuvector::SelectOp, arith::SelectOp, npuvector::CmpIOp, npuvector::CmpFOp,
            arith::MaximumFOp, arith::MinimumFOp>(op)) {
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
      if (auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp())) {
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
    int64_t n = valueType.hasStaticShape() ? valueType.getNumElements() : ShapedType::kDynamic;
    int64_t validN = n;
    if (n == ShapedType::kDynamic) {
      dynamicExtent = dynamicTileExtent(vector);
      if (!dynamicExtent) {
        return rewriter.notifyMatchFailure(op, "dynamic rank-1 store needs the tile's allocated extent");
      }
      n = *dynamicExtent;
    }
    // A static shape spells the valid element count too: a
    // 32-byte-aligned read allocates wider than it validates (2704
    // valid bf16 elements in a padded 2720-row). Lift n to the tile's
    // allocated bound and keep the valid count: the reshape below
    // conserves allocated elements, while the store must write only the
    // valid region, which the valid-extent reshape spells out.
    bool lifted = false;
    auto convertedTile = dyn_cast<pypto::TileType>(mapped(rewriter, vector).getType());
    if (convertedTile && convertedTile.getRank() == 1 &&
        llvm::none_of(convertedTile.getShape(), ShapedType::isDynamic)) {
      // The converted value carries the actual allocation. Keep the
      // source's valid count, or inherit its runtime valid region.
      n = convertedTile.getShape()[0];
      lifted = validN != ShapedType::kDynamic && n > validN;
    } else if (auto allocation = tileAllocationShape(vector);
        allocation && allocation->size() == 1 && !ShapedType::isDynamic((*allocation)[0]) && (*allocation)[0] > n) {
      n = (*allocation)[0];
      lifted = true;
    }
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (!dynamicExtent && n * elemBytes % 32 != 0) {
      // Sub-32B rows cannot hold a tile at all: fall back to
      // element-wise scalar writes.
      return storeRankOneElementWise(rewriter, loc, op, vector, destType, elemType, validN);
    }
    Value reshaped = lifted ? reshapeTileWithValidOp(rewriter, loc, vector, 1, n, 1, validN, elemType)
                            : reshapeTileOp(rewriter, loc, vector, 1, n, elemType);
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
      storeDest = rewriter
                    .create<pypto::ReshapeOp>(loc, pyptoExtentType(viewType, ValueRange()),
                                              mapped(rewriter, op.getSource()), ValueRange(), ValueRange())
                    .getResult();
    }
    llvm::append_range(offsetItems, op.getIndices());
    // `pypto.store` keeps the offsets as explicit index operands;
    // convert-pypto-to-emitpy folds the constant ones back into the
    // `pl.store` offsets list.
    rewriter.create<pypto::StoreOp>(loc, reshaped, mapped(rewriter, storeDest),
                                    mapped(rewriter, ValueRange(offsetItems)));
    return success();
  }

  /// Stores a statically-sized sub-32-byte rank-1 row element-wise through
  /// `pl.write`s: the rank-1 value spans the destination's trailing axis,
  /// so the leading offsets pass through and only the last one advances per
  /// element. `pl.write` accepts dynamic offsets, so the base need not be
  /// constant.
  static LogicalResult storeRankOneElementWise(ConversionPatternRewriter &rewriter, Location loc, TransferWriteOp op,
                                               Value vector, ShapedType destType, Type elemType, int64_t validN) {
    if (std::optional<AffineMap> perm = op.getPermutationMap(); perm && !perm->isMinorIdentity()) {
      return rewriter.notifyMatchFailure(op, "sub-32B element-wise fallback needs a minor-identity permutation");
    }
    ValueRange indices = op.getIndices();
    SmallVector<Value> leading = llvm::to_vector(indices.drop_back());
    Value base = indices.back();
    Value scalarDest = op.getSource();
    if (destType.getRank() == 1) {
      Type viewType;
      if (auto memrefType = dyn_cast<MemRefType>(destType)) {
        viewType = MemRefType::get({1, memrefType.getNumElements()}, memrefType.getElementType());
      } else {
        auto tensorType = cast<RankedTensorType>(destType);
        viewType = RankedTensorType::get({1, tensorType.getNumElements()}, tensorType.getElementType());
      }
      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *destDef = op.getSource().getDefiningOp()) {
        rewriter.setInsertionPointAfter(destDef);
      } else if (auto funcOp = op->template getParentOfType<func::FuncOp>()) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      }
      scalarDest = rewriter
                     .create<pypto::ReshapeOp>(loc, pyptoExtentType(viewType, ValueRange()),
                                               mapped(rewriter, op.getSource()), ValueRange(), ValueRange())
                     .getResult();
    }
    for (int64_t k = 0; k < validN; ++k) {
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
      if (destType.getRank() == 1) {
        elemIndices.insert(elemIndices.begin(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
      }
      elemIndices.push_back(offset);
      writeScalar(rewriter, loc, scalarDest, elemIndices, elem);
    }
    return success();
  }

  /// Lift the logical store shape to the allocation carried by its source.
  static bool refineRankReducedStoreAllocation(Value vector, ShapedType valueType, SmallVector<int64_t> &shape,
                                               ArrayRef<int64_t> valid, ConversionPatternRewriter &rewriter) {
    auto convertedTile = dyn_cast<pypto::TileType>(mapped(rewriter, vector).getType());
    if (convertedTile && convertedTile.getRank() == static_cast<int64_t>(shape.size()) &&
        llvm::none_of(convertedTile.getShape(), ShapedType::isDynamic)) {
      shape.assign(convertedTile.getShape().begin(), convertedTile.getShape().end());
      return valueType.hasStaticShape() && !llvm::equal(shape, valid);
    }
    bool lifted = false;
    if (auto allocation = tileAllocationShape(vector); allocation && allocation->size() == shape.size()) {
      for (auto [dim, alloc] : llvm::zip(shape, *allocation)) {
        if (!ShapedType::isDynamic(alloc) && alloc > dim) {
          dim = alloc;
          lifted = true;
        }
      }
    }
    return lifted;
  }

  /// Stores a rank-reduced value (destination rank > tile rank) through a
  /// `[1, ..., 1, N...]` padded reshape -- the mirror image of the reshape
  /// the load-side lowering inserts after a rank-reduced read. `pl.store`
  /// offsets and shapes share the destination's coordinate system, and
  /// PTO's tile flattening only covers ND tiles: a bare rank-1 tile fails
  /// the 2D valid-shape check, so the value is padded back to the
  /// destination's rank with singleton dims on the axes it does not span.
  /// A dynamic extent resolves to the tile's allocated bound.
  static LogicalResult storeRankReducedPadded(ConversionPatternRewriter &rewriter, Location loc, TransferWriteOp op,
                                              Value vector, ShapedType destType, ShapedType valueType) {
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
    // The value's shape spells its valid extent: a 32-byte-aligned read
    // allocates wider than it validates (a valid-2 bf16 read out of a
    // padded-16 row), so the store's reshape must name the allocated
    // bound on every dim while the store itself writes only the valid
    // region. Lift each dim to the tile's allocated bound and keep the
    // valid shape for the store's valid extents.
    SmallVector<int64_t> valid = shape;
    bool lifted = refineRankReducedStoreAllocation(vector, valueType, shape, valid, rewriter);
    // Each tile dim spans the destination dim its permutation-map result
    // projects onto; every other destination dim is padded with a 1. A
    // minor-identity map keeps the tile dims trailing, while a leading
    // projection (a transposed store) places them on the named dim. The
    // store offsets already address the destination, so they pass through
    // unchanged and only the padded shape tracks the projection.
    SmallVector<int64_t> padded(destType.getRank(), 1);
    SmallVector<int64_t> validRanked(destType.getRank(), 1);
    if (std::optional<AffineMap> perm = op.getPermutationMap()) {
      if (perm->getNumResults() != shape.size() || !perm->isProjectedPermutation()) {
        return rewriter.notifyMatchFailure(op, "rank-reduced padded store needs a projected permutation map");
      }
      for (auto [result, dim] : llvm::zip(perm->getResults(), shape)) {
        padded[cast<AffineDimExpr>(result).getPosition()] = dim;
      }
      for (auto [result, dim] : llvm::zip(perm->getResults(), valid)) {
        validRanked[cast<AffineDimExpr>(result).getPosition()] = dim;
      }
    } else {
      llvm::copy(shape, padded.end() - shape.size());
      llvm::copy(valid, validRanked.end() - valid.size());
    }
    Value reshaped;
    if (!lifted) {
      Type paddedType = pypto::TileType::get(rewriter.getContext(), elemType, padded);
      reshaped = rewriter.create<pypto::ReshapeOp>(loc, pyptoExtentType(paddedType, ValueRange()),
                                                   mapped(rewriter, vector), ValueRange(), ValueRange());
    } else {
      // A static lifted result type would trip the IR verifier's
      // element-count check, which compares against the source's valid
      // count; the dynamic type plus extents/valid-extents lowers to
      // reshape+slice, naming the allocation while narrowing the store
      // back down to the valid region. The pypto frontend cannot slice
      // an Nd tile (FlattenTileNdTo2D), so above rank 2 only the extents
      // are named: the identity reshape keeps the source view and the
      // frontend's valid-shape deduction propagates the valid region
      // through the unit axes.
      SmallVector<int64_t> dynPadded(padded.size(), ShapedType::kDynamic);
      Type paddedType = pypto::TileType::get(rewriter.getContext(), elemType, dynPadded);
      SmallVector<Value> extents;
      SmallVector<Value> validExtents;
      extents.reserve(padded.size());
      validExtents.reserve(padded.size());
      for (auto [dim, validDim] : llvm::zip(padded, validRanked)) {
        extents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
        if (padded.size() <= 2) {
          validExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, validDim));
        }
      }
      reshaped = rewriter.create<pypto::ReshapeOp>(loc, pyptoExtentType(paddedType, ValueRange(extents)),
                                                   mapped(rewriter, vector), mapped(rewriter, ValueRange(extents)),
                                                   mapped(rewriter, ValueRange(validExtents)));
    }
    rewriter.create<pypto::StoreOp>(loc, reshaped, mapped(rewriter, op.getSource()), mapped(rewriter, op.getIndices()));
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
static bool hasScalarCombine(pypto::ReduceKind kind) { return true; }

/// Reads one element out of a tile.
static Value readTileElement(ConversionPatternRewriter &rewriter, Location loc, Type resultType, Value tile,
                             ArrayRef<int64_t> index) {
  SmallVector<Value> coords;
  coords.reserve(index.size());
  llvm::transform(index, std::back_inserter(coords),
                  [&](int64_t k) { return rewriter.create<arith::ConstantIndexOp>(loc, k); });
  return rewriter
    .create<pypto::ReadOp>(loc, pyptoType(resultType), mapped(rewriter, tile), mapped(rewriter, ValueRange(coords)))
    .getResult();
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
    return rewriter.create<pypto::MinOp>(loc, pyptoType(resultType), mapped(rewriter, acc), mapped(rewriter, elem));
  }
  if (kind == pypto::ReduceKind::max) {
    return rewriter.create<pypto::MaxOp>(loc, pyptoType(resultType), mapped(rewriter, acc), mapped(rewriter, elem));
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
    // through to the full-reduction lowering below; a multi-axis `dims`
    // that covers every source axis is likewise a full reduction (the
    // vectorizer stamps e.g. [0, 1] over a rank-2 tile), not a partial one.
    if constexpr (std::is_same_v<ReductionOp, npuvector::ReductionOp>) {
      auto dims = op.getReductionDims();
      if (dims && !dims->empty() && shaped.getRank() >= 2) {
        bool coversAllAxes = dims->size() == static_cast<size_t>(shaped.getRank());
        if (coversAllAxes) {
          llvm::SmallDenseSet<int64_t> covered;
          for (int64_t d : *dims) {
            if (d < 0 || d >= static_cast<int64_t>(shaped.getRank()) || !covered.insert(d).second) {
              coversAllAxes = false;
              break;
            }
          }
        }
        if (!coversAllAxes) {
          return rewritePartialReduction(op, rewriter, *kind, shaped);
        }
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
    Type resultType = pyptoResultType(op->getResult(0));
    SmallVector<int64_t> scratchShape;
    if (shaped.hasStaticShape()) {
      scratchShape.assign(shaped.getShape().begin(), shaped.getShape().end());
    } else {
      auto bounds = tileAllocationShape(op.getVector());
      if (!bounds) {
        return rewriter.notifyMatchFailure(op, "a dynamic reduction needs the tile's allocation bounds");
      }
      scratchShape = std::move(*bounds);
    }
    if (Operation *def = op.getVector().getDefiningOp()) {
      for (Value operand : def->getOperands()) {
        if (!isa<ShapedType>(operand.getType())) {
          continue;
        }
        SmallVector<int64_t> operandBounds;
        if (auto bounds = tileAllocationShape(operand, 1)) {
          operandBounds = *bounds;
        }
        SmallVector<int64_t> downstreamBounds;
        collectSiblingAllocationBounds(operand, 0, downstreamBounds);
        if (operandBounds.empty()) {
          operandBounds = std::move(downstreamBounds);
        } else if (operandBounds.size() == downstreamBounds.size()) {
          for (unsigned axisIdx = 0; axisIdx < operandBounds.size(); ++axisIdx) {
            if (!ShapedType::isDynamic(downstreamBounds[axisIdx])) {
              operandBounds[axisIdx] = std::max(operandBounds[axisIdx], downstreamBounds[axisIdx]);
            }
          }
        }
        if (scratchShape.size() != operandBounds.size()) {
          continue;
        }
        for (unsigned axisIdx = 0; axisIdx < scratchShape.size(); ++axisIdx) {
          if (!ShapedType::isDynamic(operandBounds[axisIdx])) {
            scratchShape[axisIdx] = std::max(scratchShape[axisIdx], operandBounds[axisIdx]);
          }
        }
      }
    }
    Value tmp = createTmpTile(rewriter, loc, scratchShape, shaped.getElementType());
    Value reduced =
      rewriter.create<pypto::ReductionOp>(loc, pyptoType(resultType), kind, axis, mapped(rewriter, op.getVector()), tmp)
        .getResult();
    if (op.getAcc()) {
      reduced = rewriter.create<pypto::AddOp>(loc, pyptoType(resultType), reduced, mapped(rewriter, op.getAcc()));
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

    Type resultType = pyptoResultType(op->getResult(0));
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
      rewriter.create<pypto::ReductionOp>(loc, pyptoType(resultType), kind, pypto::ReduceAxis::row, reshaped, rowTmp)
        .getResult();
    return combineRowPartials(rewriter, loc, kind, resultType, row, r);
  }

  /// Hybrid lowering: row-reduce to [rows, 1] on device, then combine the
  /// surviving `rows` scalars.
  static Value reduceViaDeviceRowReduce(ConversionPatternRewriter &rewriter, Location loc, pypto::ReduceKind kind,
                                        ShapedType shaped, Type resultType, Value input, int64_t rows) {
    Value rowTmp = createTmpTile(rewriter, loc, shaped.getShape(), shaped.getElementType());
    Value row = rewriter
                  .create<pypto::ReductionOp>(loc, pyptoType(resultType), kind, pypto::ReduceAxis::row,
                                              mapped(rewriter, input), rowTmp)
                  .getResult();
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

/// Raises `bound` onto `siblingBounds[axis]`, initializing the bounds vector
/// on first use and keeping the per-axis maximum.
static void raiseSiblingBound(SmallVectorImpl<int64_t> &siblingBounds, unsigned rank, unsigned axis, int64_t bound) {
  if (siblingBounds.empty()) {
    siblingBounds.assign(rank, ShapedType::kDynamic);
  }
  siblingBounds[axis] =
    ShapedType::isDynamic(siblingBounds[axis]) ? bound : std::max(siblingBounds[axis], bound);
}

/// The sibling allocation: PTOAS lays every tile in the block onto one
/// grid, so the carrier/constant a broadcast materializes must also
/// cover the allocation of the operands its result merges with -- e.g.
/// a row grid the load padded from 139 to 140 -- or the expand/add
/// that combines them mismatches at compile time. Only the operands
/// sitting next to this broadcast inside its direct consumers carry
/// that grid; the values feeding those operands resolve through
/// `tileAllocationShape` (loads, carriers, lifted views, expands).
template <typename BroadcastOp>
static void collectBroadcastSiblingBounds(BroadcastOp op, unsigned rank, SmallVectorImpl<int64_t> &siblingBounds) {
  for (Operation *user : op->getResult(0).getUsers()) {
    for (Value operand : user->getOperands()) {
      if (operand == op->getResult(0) || !isa<ShapedType>(operand.getType())) {
        continue;
      }
      auto shape = tileAllocationShape(operand, 1);
      if (!shape || shape->size() != rank) {
        continue;
      }
      for (unsigned axis = 0; axis < rank; ++axis) {
        if (ShapedType::isDynamic((*shape)[axis])) {
          continue;
        }
        raiseSiblingBound(siblingBounds, rank, axis, (*shape)[axis]);
      }
    }
  }
}

/// A rank-1 tile lifted onto a rank-2 result materializes as an
/// expand of the reshaped source onto this zero carrier, and that
/// reshape spells the allocated extent the source's read padded it
/// to (9 fp32 elements align to 16). The carrier must cover the
/// same extent on the identity-mapped axis -- also exactly what
/// `tileAllocationShape`'s lift branch reports to sibling queries
/// -- or the expand combines a [9, cols] carrier with a [16, 1]
/// vector and every downstream tile mismatches at compile time.
template <typename BroadcastOp>
static void liftBroadcastIdentityBound(BroadcastOp op, unsigned rank, SmallVectorImpl<int64_t> &siblingBounds) {
  if (rank != 2 || op.getDimension().size() != 1 || op.getDimension()[0] < 0 ||
      op.getDimension()[0] >= static_cast<int64_t>(rank)) {
    return;
  }
  auto srcType = dyn_cast<npuvector::NPUVectorType>(op.getSource().getType());
  if (!srcType || srcType.getRank() != 1) {
    return;
  }
  auto srcAlloc = tileAllocationShape(op.getSource());
  if (!srcAlloc || srcAlloc->size() != 1 || ShapedType::isDynamic((*srcAlloc)[0])) {
    return;
  }
  raiseSiblingBound(siblingBounds, rank, static_cast<unsigned>(op.getDimension()[0]), (*srcAlloc)[0]);
}

/// Pads a constant allocation bound for one result axis of a dynamic
/// broadcast: the contiguous axis and the axes surviving a downstream
/// partial reduction pad (by that reduction's result dtype), matching the
/// extent a dynamic transfer_read allocates, so broadcast tiles stay
/// shape-compatible with the loads they combine with; other leading axes
/// stay unpadded for the same reason the read side leaves them alone.
/// The sibling grid raises the bound the same way.
static int64_t padBroadcastAxisBound(int64_t bound, unsigned axis, unsigned rank, unsigned elemBytes,
                                     const SmallVectorImpl<ReductionAxisPad> &pads,
                                     const SmallVectorImpl<int64_t> &siblingBounds) {
  int64_t padded = padAllocationExtent(bound, axis, rank, elemBytes, pads);
  if (axis < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[axis])) {
    padded = std::max(padded, siblingBounds[axis]);
  }
  return padded;
}

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
    SmallVector<ReductionAxisPad> pads;
    collectReductionAxisPads(op->getResult(0), 0, pads);
    SmallVector<int64_t> siblingBounds;
    collectBroadcastSiblingBounds(op, rank, siblingBounds);
    liftBroadcastIdentityBound(op, rank, siblingBounds);
    unsigned elemBytes = resultType.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned dynIdx = 0;
    for (auto [axis, dim] : llvm::enumerate(resultType.getShape())) {
      if (dim == ShapedType::kDynamic) {
        Value alloc = perDim ? (axis < maxSizes.size() ? maxSizes[axis] : dynamicSizes[axis])
                             : (dynIdx < maxSizes.size() ? maxSizes[dynIdx] : dynamicSizes[dynIdx]);
        if (auto bound = constIndexValue(alloc)) {
          int64_t padded = padBroadcastAxisBound(*bound, axis, rank, elemBytes, pads, siblingBounds);
          if (padded != *bound) {
            alloc = rewriter.create<arith::ConstantIndexOp>(loc, padded);
          }
        }
        shapeValues.push_back(alloc);
        ++dynIdx;
      } else {
        // A static axis surviving a downstream reduction pads too: the
        // allocation must cover the reduction's result rows. The
        // sibling grid raises it the same way.
        int64_t padded = padBroadcastAxisBound(dim, axis, rank, elemBytes, pads, siblingBounds);
        shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, padded));
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

/// The outbound reshape of a row expand targets a rank-N destination, but
/// the flat carrier keeps one extent per rank-2 axis: `pypto.reshape`
/// takes one extent and one valid bound per result dimension. The leading
/// axes keep their destination extent (their product is the carrier's row
/// count), the last axis carries the 32B-aligned carrier columns, and the
/// valid window is the destination shape throughout.
static void appendRowExpandResultExtents(const SmallVector<int64_t> &destShape, const SmallVector<Value> &destBounds,
                                         bool dynamicCols, int64_t carrierCols, unsigned rank, Location loc,
                                         SmallVectorImpl<Value> &resultExtents, SmallVectorImpl<Value> &resultValid,
                                         ConversionPatternRewriter &rewriter) {
  for (unsigned axis = 0; axis + 1 < rank; ++axis) {
    if (!ShapedType::isDynamic(destShape[axis])) {
      resultExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, destShape[axis]));
      resultValid.push_back(rewriter.create<arith::ConstantIndexOp>(loc, destShape[axis]));
    } else {
      resultExtents.push_back(destBounds[axis]);
      resultValid.push_back(destBounds[axis]);
    }
  }
  if (!dynamicCols) {
    resultExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, carrierCols));
    resultValid.push_back(rewriter.create<arith::ConstantIndexOp>(loc, destShape.back()));
  } else {
    resultExtents.push_back(destBounds.back());
    resultValid.push_back(destBounds.back());
  }
}

/// Row-expands `vec` — whose broadcast axis (the dropped one) already
/// sits last with extent 1 — against a zero carrier: both collapse
/// their alignment axes onto a rank-2 [rows, 1] / [rows, cols] pair,
/// the expand family's vector contract, and the result reshapes back
/// to `destShape`. The leading extents of `vecShape` and `destShape`
/// agree — they differ only in the broadcast axis.
static Value rowExpandOntoCarrier(ShapedType shapedResult, Location loc, StringRef zero, unsigned rank, Value vec,
                                  const SmallVector<int64_t> &vecShape, const SmallVector<Value> &vecBounds,
                                  const SmallVector<int64_t> &vecAlloc, const SmallVector<int64_t> &destShape,
                                  const SmallVector<Value> &destBounds, ConversionPatternRewriter &rewriter) {
  auto constant = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
  unsigned elemBytes = std::max<unsigned>(1, shapedResult.getElementType().getIntOrFloatBitWidth() / 8);
  int64_t staticRows = 1;
  bool dynamicRows = false;
  for (unsigned axis = 0; axis + 1 < rank; ++axis) {
    if (ShapedType::isDynamic(vecShape[axis]))
      dynamicRows = true;
    else
      staticRows *= vecShape[axis];
  }
  Value rowsExtent;
  auto rows = [&]() -> Value {
    if (!dynamicRows)
      return constant(staticRows);
    if (!rowsExtent) {
      rowsExtent = constant(1);
      for (unsigned axis = 0; axis + 1 < rank; ++axis) {
        Value dim = vecBounds.empty() ? constant(vecShape[axis]) : vecBounds[axis];
        rowsExtent = rewriter.create<arith::MulIOp>(loc, rowsExtent, dim);
      }
    }
    return rowsExtent;
  };

  // The source may carry a padded trailing identity axis. After the ferry,
  // the dropped singleton is last, so preserve the padding on the axis
  // immediately before it while flattening the physical allocation. The
  // ferry reorders the tracked shape, so the allocation shape arrives
  // reordered the same way and keeps matching it axis by axis.
  SmallVector<int64_t> vecAllocShape = vecAlloc;
  if (!dynamicRows && rank > 1) {
    unsigned paddedAxis = rank - 1;
    if (vecShape.back() == 1)
      paddedAxis = rank - 2;
    vecAllocShape[paddedAxis] = alignExtentTo32B(vecAllocShape[paddedAxis], elemBytes);
  }
  int64_t allocProduct = std::accumulate(vecAllocShape.begin(), vecAllocShape.end(), static_cast<int64_t>(1),
                                         std::multiplies<int64_t>());
  SmallVector<int64_t> vecFlatShape = {dynamicRows ? ShapedType::kDynamic : staticRows, 1};
  SmallVector<Value> vecFlatExtents;
  SmallVector<Value> vecFlatValid;
  if (dynamicRows) {
    vecFlatExtents.push_back(rows());
    vecFlatExtents.push_back(constant(1));
  } else {
    vecFlatExtents.push_back(constant(allocProduct));
    vecFlatExtents.push_back(constant(1));
    vecFlatValid.push_back(constant(staticRows));
    vecFlatValid.push_back(constant(1));
  }
  Value vecFlat = rewriter.create<pypto::ReshapeOp>(
    loc, pyptoExtentType(pyptoType(shapedResult.clone(vecFlatShape)), ValueRange(vecFlatExtents)),
    mapped(rewriter, vec), mapped(rewriter, ValueRange(vecFlatExtents)), mapped(rewriter, ValueRange(vecFlatValid)));

  bool dynamicCols = ShapedType::isDynamic(destShape.back());
  int64_t carrierCols = dynamicCols ? 0 : alignExtentTo32B(destShape.back(), elemBytes);
  // The carrier's result type spells the valid window ([rows, destCols])
  // so the outbound reshape conserves elements across types; the
  // 32B-aligned allocation rides in `tile_full`'s extents operands,
  // which the emitpy lowering forwards verbatim as the `pl.tile.full`
  // shape list. `RowExpandAdd` keeps the carrier's type, so the
  // expanded tile carries the same valid/alloc split.
  SmallVector<int64_t> carrierFlatShape = vecFlatShape;
  carrierFlatShape.back() = dynamicCols ? ShapedType::kDynamic : destShape.back();
  SmallVector<Value> carrierFlatBounds;
  if (dynamicRows || dynamicCols) {
    carrierFlatBounds.push_back(rows());
    carrierFlatBounds.push_back(dynamicCols ? destBounds.back() : constant(carrierCols));
  }
  Value carrier =
    rewriter
      .create<pypto::TileFullOp>(
        loc, pyptoExtentType(pyptoType(shapedResult.clone(carrierFlatShape)), ValueRange(carrierFlatBounds)),
        rewriter.getStringAttr(zero), mapped(rewriter, ValueRange(carrierFlatBounds)))
      .getResult();
  Value expanded =
    rewriter.create<pypto::RowExpandAddOp>(loc, pyptoType(shapedResult.clone(carrierFlatShape)), carrier, vecFlat)
      .getResult();
  SmallVector<Value> resultExtents;
  SmallVector<Value> resultValid;
  appendRowExpandResultExtents(destShape, destBounds, dynamicCols, carrierCols, rank, loc, resultExtents, resultValid,
                               rewriter);
  return rewriter
    .create<pypto::ReshapeOp>(loc, pyptoExtentType(pyptoType(shapedResult.clone(destShape)), ValueRange(resultExtents)),
                              expanded, mapped(rewriter, ValueRange(resultExtents)),
                              mapped(rewriter, ValueRange(resultValid)))
    .getResult();
}

/// Lifts one dropped axis of a general lift. The expand family only
/// broadcasts the last two dimensions and its vector operand contract
/// is rank-2 ([rows, 1] for a row expand, [1, cols] for a col expand),
/// so a dropped axis outside them is ferried to the tail, the
/// alignment axes ahead of the broadcast one are collapsed with
/// `pypto.reshape` onto a rank-2 column vector, lifted with a row
/// expand against a zero carrier, and restored. The dropped axis is
/// always a singleton, so the outbound ferry reorders only the
/// tracked shape — a singleton swap leaves the linear element order
/// untouched — while the restore, whose axis now carries the real
/// extent, keeps its `pypto.transpose` swaps.
static Value liftDroppedAxis(ShapedType shapedResult, Location loc, StringRef zero, unsigned rank, unsigned dropped,
                             Value cur, const SmallVector<int64_t> &curShape, const SmallVector<int64_t> &curAllocShape,
                             const SmallVector<Value> &curBounds, const SmallVector<int64_t> &targetShape,
                             const SmallVector<Value> &targetBounds, ConversionPatternRewriter &rewriter) {
  auto swapAxes = [&](SmallVector<int64_t> &shape, SmallVector<Value> &bounds, Value &value, unsigned a, unsigned b) {
    std::swap(shape[a], shape[b]);
    if (!bounds.empty()) {
      std::swap(bounds[a], bounds[b]);
    }
    value = rewriter.create<pypto::TransposeOp>(loc, pyptoType(shapedResult.clone(shape)), mapped(rewriter, value),
                                                static_cast<uint64_t>(a), static_cast<uint64_t>(b));
  };
  if (rank == 2) {
    // A rank-2 working shape already meets the expand family's vector
    // contract — [rows, 1] for the last axis, [1, cols] for the
    // second-to-last — so the dropped axis broadcasts directly against
    // a zero carrier.
    auto curShaped = cast<ShapedType>(cur.getType());
    Type expandType = curShaped.clone(targetShape);
    Value carrier =
      rewriter
        .create<pypto::TileFullOp>(loc, pyptoExtentType(expandType, ValueRange(targetBounds)),
                                   rewriter.getStringAttr(zero), mapped(rewriter, ValueRange(targetBounds)))
        .getResult();
    Value expanded =
      dropped + 1 == rank
        ? rewriter.create<pypto::RowExpandAddOp>(loc, pyptoType(expandType), carrier, mapped(rewriter, cur)).getResult()
        : rewriter.create<pypto::ColExpandAddOp>(loc, pyptoType(expandType), carrier, mapped(rewriter, cur))
            .getResult();
    if (expandType != shapedResult.clone(targetShape)) {
      expanded = rewriter.create<pypto::ReinterpretViewOp>(loc, pyptoType(shapedResult.clone(targetShape)), expanded)
                   .getResult();
    }
    return expanded;
  }
  if (dropped + 1 == rank) {
    // Tail position: the dropped axis already sits last, so only the
    // alignment axes ahead of it need collapsing.
    return rowExpandOntoCarrier(shapedResult, loc, zero, rank, cur, curShape, curBounds, curAllocShape, targetShape,
                                targetBounds, rewriter);
  }
  // Ferry the dropped axis to the tail, row-expand there, and
  // ferry back. The dropped axis is always a singleton, so ferrying
  // it is an identity permutation on the linear element order: the
  // outbound `pypto.transpose` swaps are elided and only the tracked
  // shape/bounds reordered — the flatten reshape reads the same
  // linear buffer either way. The allocation shape ferries along so
  // the flatten still bounds-checks the same physical tile.
  SmallVector<int64_t> vecShape = curShape;
  SmallVector<Value> vecBounds = curBounds;
  SmallVector<int64_t> vecAlloc = curAllocShape;
  for (unsigned axis = dropped; axis + 1 < rank; ++axis) {
    std::swap(vecShape[axis], vecShape[axis + 1]);
    std::swap(vecAlloc[axis], vecAlloc[axis + 1]);
    if (!vecBounds.empty()) {
      std::swap(vecBounds[axis], vecBounds[axis + 1]);
    }
  }
  Value vec = cur;
  SmallVector<int64_t> destShape = targetShape;
  SmallVector<Value> destBounds = targetBounds;
  for (unsigned axis = dropped; axis + 1 < rank; ++axis) {
    std::swap(destShape[axis], destShape[axis + 1]);
    if (!destBounds.empty()) {
      std::swap(destBounds[axis], destBounds[axis + 1]);
    }
  }
  Value expanded = rowExpandOntoCarrier(shapedResult, loc, zero, rank, vec, vecShape, vecBounds, vecAlloc, destShape,
                                        destBounds, rewriter);
  for (unsigned axis = rank - 1; axis-- > dropped;) {
    swapAxes(destShape, destBounds, expanded, axis, axis + 1);
  }
  return expanded;
}

/// Per-axis runtime bounds for a general lift: a dropped axis takes the
/// broadcast's aligned extent (the extent the lift expands it to), an
/// identity-mapped axis the source tile's allocated extent — the opening
/// reshape conserves the source's physical elements, so its view and
/// every transposed or expanded derivative of it must keep the padded
/// bound the source's producing read or reduction allocated, never the
/// broadcast's aligned max size (a valid bound). The max size stands in
/// when the allocation is not statically resolvable.
static SmallVector<Value> liftAxisBounds(const SmallVector<bool> &isDropped, unsigned rank,
                                         npuvector::NPUVectorType srcType,
                                         const std::optional<SmallVector<int64_t>> &srcAlloc,
                                         const SmallVectorImpl<Value> &extents, Location loc,
                                         ConversionPatternRewriter &rewriter) {
  auto constant = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
  SmallVector<Value> axisBounds(rank);
  unsigned boundSourceDim = 0;
  for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
    if (isDropped[resultDim] || !ShapedType::isDynamic(srcType.getDimSize(boundSourceDim)) || !srcAlloc ||
        boundSourceDim >= srcAlloc->size()) {
      axisBounds[resultDim] = extents[resultDim];
    } else {
      axisBounds[resultDim] = constant((*srcAlloc)[boundSourceDim]);
    }
    if (!isDropped[resultDim]) {
      ++boundSourceDim;
    }
  }
  return axisBounds;
}

/// A statically-shaped working view over a source whose allocated shape
/// resolves past it conserves the source's physical elements: the reshape
/// views the allocation (extents, e.g. [1,8,32] keeping a padded cmps'
/// 256 elements) while the valid extents name the working shape, so the
/// slice the lowering renders keeps every downstream ferry/expand on the
/// valid region alone — the same conservation the dynamic-shape extents
/// spell through the working shape's bounds. An unpadded or unresolvable
/// source keeps the plain reshape.
static void appendLiftConservedExtents(const SmallVector<bool> &isDropped, unsigned rank, unsigned srcRank,
                                       const std::optional<SmallVector<int64_t>> &srcAlloc,
                                       const SmallVector<int64_t> &curShape, Location loc,
                                       SmallVectorImpl<Value> &curExtents, SmallVectorImpl<Value> &curValid,
                                       ConversionPatternRewriter &rewriter) {
  if (llvm::none_of(curShape, [](int64_t d) { return ShapedType::isDynamic(d); }) && srcAlloc &&
      srcAlloc->size() == srcRank &&
      llvm::none_of(*srcAlloc, [](int64_t d) { return ShapedType::isDynamic(d); })) {
    SmallVector<int64_t> allocShape(rank);
    int64_t allocProduct = 1;
    int64_t validProduct = 1;
    unsigned allocSourceDim = 0;
    for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
      allocShape[resultDim] = isDropped[resultDim] ? 1 : (*srcAlloc)[allocSourceDim++];
      allocProduct *= allocShape[resultDim];
      validProduct *= curShape[resultDim];
    }
    if (allocProduct > validProduct) {
      for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
        curExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, allocShape[resultDim]));
        curValid.push_back(rewriter.create<arith::ConstantIndexOp>(loc, curShape[resultDim]));
      }
    }
  }
}

/// General lift onto any rank: `dimension` maps every source axis
/// onto an identity-matched result axis (in increasing order) and
/// every remaining axis is dropped — the lift may span any rank
/// jump and drop axes anywhere. The expand family only broadcasts
/// the last two dimensions, so a dropped axis outside them is
/// ferried to the tail, lifted with a row expand against a zero
/// carrier, and ferried back (the outbound ferry is shape-only,
/// the dropped axis being a singleton).
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
  Type convertedResult = pyptoResultType(op->getResult(0));
  FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
  if (failed(extents)) {
    return failure();
  }
  auto shapedResult = cast<ShapedType>(convertedResult);
  auto constant = [&](int64_t v) { return rewriter.create<arith::ConstantIndexOp>(loc, v); };
  auto srcAlloc = tileAllocationShape(op.getSource());
  SmallVector<Value> axisBounds = liftAxisBounds(isDropped, rank, srcType, srcAlloc, *extents, loc, rewriter);
  // Per-dimension runtime bounds of a working shape, one value per
  // axis; an all-static shape carries none, mirroring the pypto
  // ops' empty-extents convention.
  auto makeBounds = [&](const SmallVector<int64_t> &shape) -> SmallVector<Value> {
    if (llvm::none_of(shape, [](int64_t d) { return ShapedType::isDynamic(d); })) {
      return {};
    }
    SmallVector<Value> bounds;
    for (auto [axis, dim] : llvm::enumerate(shape)) {
      bounds.push_back(ShapedType::isDynamic(dim) ? axisBounds[axis] : constant(dim));
    }
    return bounds;
  };
  // Start from the source reshaped with a singleton at every dropped
  // axis; identity axes carry the source extents over.
  SmallVector<int64_t> curShape;
  SmallVector<int64_t> curAllocShape;
  unsigned sourceDim = 0;
  for (unsigned resultDim = 0; resultDim < rank; ++resultDim) {
    curShape.push_back(isDropped[resultDim] ? 1 : srcType.getDimSize(sourceDim));
    curAllocShape.push_back(
      isDropped[resultDim] ? 1 : (srcAlloc ? (*srcAlloc)[sourceDim] : srcType.getDimSize(sourceDim)));
    if (!isDropped[resultDim])
      ++sourceDim;
  }
  SmallVector<Value> curBounds = makeBounds(curShape);
  SmallVector<Value> curExtents;
  SmallVector<Value> curValid;
  appendLiftConservedExtents(isDropped, rank, srcRank, srcAlloc, curShape, loc, curExtents, curValid, rewriter);
  Value cur = rewriter.create<pypto::ReshapeOp>(
    loc,
    pyptoExtentType(pyptoType(shapedResult.clone(curShape)),
                    curExtents.empty() ? ValueRange(curBounds) : ValueRange(curExtents)),
    mapped(rewriter, op.getSource()), curExtents.empty() ? ValueRange(curBounds) : ValueRange(curExtents),
    mapped(rewriter, ValueRange(curValid)));
  std::string zero = isa<FloatType>(resultType.getElementType()) ? "0.0" : "0";
  // Lift one dropped axis at a time, tail first so every expand only
  // meets already-lifted axes behind the one it broadcasts.
  for (auto droppedIt = droppedAxes.rbegin(); droppedIt != droppedAxes.rend(); ++droppedIt) {
    unsigned dropped = *droppedIt;
    SmallVector<int64_t> targetShape = curShape;
    targetShape[dropped] = resultType.getDimSize(dropped);
    SmallVector<Value> targetBounds = makeBounds(targetShape);
    cur = liftDroppedAxis(shapedResult, loc, zero, rank, dropped, cur, curShape, curAllocShape, curBounds, targetShape,
                          targetBounds, rewriter);
    curShape = std::move(targetShape);
    curAllocShape = curShape;
    curBounds = std::move(targetBounds);
  }
  rewriter.replaceOp(op, cur);
  return success();
}

/// Collects the per-dimension valid extents of a dynamic broadcast
/// result: each dynamic dimension names its `dynamicSizes` entry (under
/// either arity convention — one entry per dimension or one per `?`),
/// each static dimension its type extent. An empty result means the
/// whole result is statically fully valid, so nothing needs narrowing.
static SmallVector<Value> broadcastValidExtents(npuvector::BroadcastOp op, ShapedType resultType, Location loc,
                                                ConversionPatternRewriter &rewriter) {
  SmallVector<Value> validExtents;
  if (resultType.hasStaticShape()) {
    return validExtents;
  }
  ValueRange dynamicSizes = op.getDynamicSizes();
  unsigned rank = resultType.getRank();
  unsigned numDynamic = llvm::count(resultType.getShape(), ShapedType::kDynamic);
  bool perDim = dynamicSizes.size() == rank && rank != numDynamic;
  unsigned dynamicIndex = 0;
  for (auto [axis, dim] : llvm::enumerate(resultType.getShape())) {
    if (ShapedType::isDynamic(dim)) {
      validExtents.push_back(dynamicSizes[perDim ? axis : dynamicIndex++]);
    } else {
      validExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
    }
  }
  return validExtents;
}

/// PTOAS lays every tile in the block onto one grid, so the constant or
/// carrier a scalar broadcast materializes must also allocate the padded
/// shape of the siblings its result merges with downstream -- e.g. a
/// [3060] one against a Taylor chain over a load that pads to [3064] --
/// or the merge mismatches at compile time, the same sibling raise
/// `ArithConstantPattern` spells for splat constants. A static result
/// spells no extents, so the raise materializes them per axis; a dynamic
/// result already carries the sibling grid through `buildBroadcastExtents`.
///
/// The raised `tile_full` allocates the padded sibling shape while the
/// broadcast's own result type stays the valid contract. Name that shape
/// as the valid extents so the re-view below restores it: a consumer
/// that pairs the result with a loaded sibling -- e.g. `tile.select`,
/// whose branches must agree on valid extents -- merges tiles that all
/// carry the IR shape as their valid region, never the raised
/// allocation.
static void raiseStaticSiblingPad(Value result, ShapedType resultType, Location loc, SmallVectorImpl<Value> &extents,
                                  SmallVectorImpl<Value> &validExtents, ConversionPatternRewriter &rewriter) {
  if (!extents.empty() || !resultType.hasStaticShape()) {
    return;
  }
  SmallVector<int64_t> siblingBounds;
  collectSiblingAllocationBounds(result, 0, siblingBounds);
  if (siblingBounds.size() != static_cast<size_t>(resultType.getRank())) {
    return;
  }
  bool needsPad = false;
  for (auto [axis, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(siblingBounds[axis]) && siblingBounds[axis] > dim) {
      needsPad = true;
      break;
    }
  }
  if (!needsPad) {
    return;
  }
  for (auto [axis, dim] : llvm::enumerate(resultType.getShape())) {
    int64_t size = dim;
    if (!ShapedType::isDynamic(siblingBounds[axis])) {
      size = std::max(size, siblingBounds[axis]);
    }
    extents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, size));
  }
  std::transform(resultType.getShape().begin(), resultType.getShape().end(), std::back_inserter(validExtents),
                 [&](int64_t dim) { return rewriter.create<arith::ConstantIndexOp>(loc, dim); });
}

/// The reshaped vector of a rank-1→rank-2 lift: [rows, 1] for a row
/// lift, [1, cols] for a column lift; the lifted extent names the source
/// tile's allocated extent, falling back to an aligned bound when the
/// allocation is not statically resolvable.
static std::pair<Type, SmallVector<Value>> liftedVectorView(npuvector::BroadcastOp op,
                                                            npuvector::NPUVectorType srcType, int64_t srcExtent,
                                                            unsigned axis, const SmallVectorImpl<Value> &extents,
                                                            bool rowLift, Location loc,
                                                            ConversionPatternRewriter &rewriter) {
  Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  Value extent;
  if (ShapedType::isDynamic(srcExtent)) {
    if (auto shape = tileAllocationShape(op.getSource()); shape && shape->size() == 1) {
      extent = rewriter.create<arith::ConstantIndexOp>(loc, (*shape)[0]);
    } else {
      extent = extents[axis];
    }
  } else {
    // Static source extent: the allocated bound the view must match.
    unsigned elemBytes = srcType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t bound;
    if (auto shape = tileAllocationShape(op.getSource()); shape && shape->size() == 1) {
      bound = (*shape)[0];
    } else {
      bound = alignExtentTo32B(srcExtent, elemBytes);
    }
    extent = rewriter.create<arith::ConstantIndexOp>(loc, bound);
  }
  Type vecType;
  SmallVector<Value> vecExtents;
  if (rowLift) {
    vecType = srcType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
    vecExtents.push_back(extent);
    vecExtents.push_back(one);
  } else {
    vecType = srcType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
    vecExtents.push_back(one);
    vecExtents.push_back(extent);
  }
  return {vecType, vecExtents};
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
/// The reshaped vector's lifted extent names the source tile's
/// allocated extent — `pl.reshape` conserves physical elements, so the
/// view must match the padded bound the source's producing read or
/// reduction allocated, never the broadcast's aligned max size (a
/// valid bound); the max size stands in when the allocation is not
/// statically resolvable. A static source extent is a valid bound too:
/// the view carries the allocated extent as a constant the emitpy
/// lowering renders literally. Target types record that physical extent;
/// runtime bounds remain dynamic. The carrier
/// keeps the aligned bound `buildBroadcastExtents` produces — PTOAS
/// pads a `tile_full` allocation onto the same row grid, so the
/// expand still combines same-shaped tiles. The identity-mapped axis
/// must agree on extent and staticality, or a constant source could
/// be reshaped past its allocation bound.
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
  Type convertedResult = pyptoResultType(op->getResult(0));
  FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
  if (failed(extents)) {
    return failure();
  }
  // The reshaped vector: [rows, 1] for a row lift, [1, cols] for a
  // column lift; the lifted extent names the source tile's allocated
  // extent, falling back to an aligned bound when the allocation is
  // not statically resolvable.
  auto [vecType, vecExtents] = liftedVectorView(op, srcType, srcExtent, axis, *extents, rowLift, loc, rewriter);
  // The reshaped vector stays a bare reshape: its extents spell the
  // allocated `[rows, 1]` bound literally, so the runtime derives a
  // clean `[srcValid, 1]` valid region (the unit-axis rank change) and
  // reads the column-major layout a static `[N, 1]` literal implies —
  // exactly the src1 contract the expand families check for. Naming
  // valid extents instead would slice a dynamic `[?, 1]` view whose
  // row-major layout fails the packed-column check.
  Value vec = rewriter.create<pypto::ReshapeOp>(loc, pyptoExtentType(vecType, ValueRange(vecExtents)),
                                                mapped(rewriter, op.getSource()),
                                                mapped(rewriter, ValueRange(vecExtents)), ValueRange());
  std::string zero = isa<FloatType>(resultType.getElementType()) ? "0.0" : "0";
  Value carrier = rewriter
                    .create<pypto::TileFullOp>(loc, pyptoExtentType(convertedResult, ValueRange(*extents)),
                                               rewriter.getStringAttr(zero), mapped(rewriter, ValueRange(*extents)))
                    .getResult();
  SmallVector<Value> carrierValid = broadcastValidExtents(op, resultType, loc, rewriter);
  Value result;
  if (rowLift) {
    // A fresh `tile_full` is valid everywhere, but row expand demands
    // provably equal valid extents between the carrier and the reshaped
    // vector. Re-view the carrier with the broadcast's own dynamic sizes
    // — the values the source's valid extents trace to — so the runtime's
    // row count proof equates the two; the emitpy lowering renders the
    // re-view as a reshape+slice sandwich.
    if (!carrierValid.empty()) {
      carrier =
        rewriter
          .create<pypto::ReshapeOp>(loc, pyptoExtentType(convertedResult, ValueRange(*extents)), carrier,
                                    mapped(rewriter, ValueRange(*extents)), mapped(rewriter, ValueRange(carrierValid)))
          .getResult();
    }
    result = rewriter.create<pypto::RowExpandAddOp>(loc, pyptoType(convertedResult), carrier, mapped(rewriter, vec))
               .getResult();
  } else {
    // Column expand mirrors that with one asymmetry: its codegen cannot
    // read a pto.subview operand and materializes one into a pto.textract,
    // whose element type A5 restricts to the fp8/f16/bf16/f32 and int8
    // families — an i32 carrier dies right there. Feed the expand the
    // bare tile_full (zeros everywhere, so the rows past the valid region
    // just carry replicated columns) and re-view the expand's fresh
    // result instead: consumers outside the tcolexpand family read
    // subview SSAs natively.
    result = rewriter.create<pypto::ColExpandAddOp>(loc, pyptoType(convertedResult), carrier, mapped(rewriter, vec))
               .getResult();
    if (!carrierValid.empty()) {
      result =
        rewriter
          .create<pypto::ReshapeOp>(loc, pyptoExtentType(convertedResult, ValueRange(*extents)), result,
                                    mapped(rewriter, ValueRange(*extents)), mapped(rewriter, ValueRange(carrierValid)))
          .getResult();
    }
  }
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
    Type convertedResult = pyptoResultType(op->getResult(0));

    // The tile's allocated extent: static dims stay in the result type;
    // dynamic dims take their `maxSizes` upper bound (constants fold into
    // index constants, runtime bounds stay operand values).
    FailureOr<SmallVector<Value>> extents = buildBroadcastExtents(op, resultType, loc, rewriter);
    if (failed(extents)) {
      return failure();
    }
    SmallVector<Value> validExtents;
    if constexpr (std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
      validExtents = broadcastValidExtents(op, resultType, loc, rewriter);
    }

    // PTOAS lays every tile in the block onto one grid, so the constant or
    // carrier a scalar broadcast materializes must also allocate the padded
    // shape of the siblings its result merges with downstream; a static
    // result spells no extents, so the raise materializes them per axis.
    raiseStaticSiblingPad(op.getResult(), resultType, op.getLoc(), *extents, validExtents, rewriter);
    convertedResult = pyptoExtentType(convertedResult, *extents);

    auto materializeFull = [&](StringRef value) -> Value {
      Value full = rewriter
                     .create<pypto::TileFullOp>(loc, pyptoExtentType(convertedResult, ValueRange(*extents)),
                                                rewriter.getStringAttr(value), mapped(rewriter, ValueRange(*extents)))
                     .getResult();
      if (validExtents.empty()) {
        return full;
      }
      return rewriter
        .create<pypto::ReshapeOp>(loc, pyptoExtentType(convertedResult, *extents), full, ValueRange(*extents),
                                  ValueRange(validExtents))
        .getResult();
    };

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
      result = materializeFull(*constText);
    } else {
      std::string zero = isa<FloatType>(sourceType) ? "0.0" : "0";
      Value carrier = materializeFull(zero);
      result = rewriter.create<pypto::AddOp>(loc, pyptoType(convertedResult), carrier, mapped(rewriter, op.getSource()))
                 .getResult();
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
      if (!isa<pypto::TileType>(mapped(rewriter, tile).getType())) {
        return rewriter.notifyMatchFailure(op, "expand data producer has not been converted yet");
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
    // The reshape stays bare like `materializeLiftedBroadcast`: its
    // extents spell the allocated bound literally, so the runtime derives
    // a clean `[srcValid, 1]` valid region (the unit-axis rank change)
    // and reads the column-major layout a static `[N, 1]` literal implies
    // — the src1 contract the expand families check for. Naming valid
    // extents instead would degrade into a reshape+slice whose window
    // inference materializes `min(srcValid, bound)`, and the expand's
    // row/column count proof could no longer equate the two sides.
    Value vec = rewriter.create<pypto::ReshapeOp>(
      op.getLoc(), pyptoExtentType(lifted->first, ValueRange(lifted->second)), mapped(rewriter, op.getSource()),
      mapped(rewriter, ValueRange(lifted->second)), ValueRange());
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

  /// The broadcast may have been materialized before the consumer's data
  /// producer was converted. Fuse that consumer once its adaptor carries
  /// the data Tile, using the same view and expand builders as above.
  static bool tryFuseConsumer(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter) {
    if (operands.size() != 2) {
      return false;
    }
    for (unsigned broadcastIndex : {1u, 0u}) {
      auto broadcast = op->getOperand(broadcastIndex).getDefiningOp<npuvector::BroadcastOp>();
      if (!broadcast) {
        continue;
      }
      ArrayRef<int64_t> dimension = broadcast.getDimension();
      bool rowLift = dimension.size() == 1 && dimension[0] == 0;
      bool colLift = dimension.size() == 1 && dimension[0] == 1;
      auto srcType = dyn_cast<npuvector::NPUVectorType>(broadcast.getSource().getType());
      Value tile = fusibleTileOperand(op, broadcast.getResult());
      auto dataType = dyn_cast<pypto::TileType>(operands[1 - broadcastIndex].getType());
      auto mappedSrcType = dyn_cast<pypto::TileType>(mapped(rewriter, broadcast.getSource()).getType());
      if ((!rowLift && !colLift) || !srcType || srcType.getRank() != 1 || !tile || !dataType ||
          dataType.getRank() != 2 || !mappedSrcType || mappedSrcType.getRank() != 1) {
        continue;
      }
      SmallVector<std::pair<Operation *, Value>> consumers{{op, tile}};
      if (failed(checkLiftedExtents(rewriter, consumers, rowLift, srcType.getDimSize(0)))) {
        continue;
      }
      FailureOr<std::pair<Type, SmallVector<Value>>> lifted =
        buildLiftedVector(broadcast, srcType, rowLift, rewriter);
      if (failed(lifted)) {
        continue;
      }
      Value vec = rewriter.create<pypto::ReshapeOp>(
        broadcast.getLoc(), pyptoExtentType(lifted->first, ValueRange(lifted->second)),
        mapped(rewriter, broadcast.getSource()), mapped(rewriter, ValueRange(lifted->second)), ValueRange());
      rewriter.replaceOp(op, makeExpand(op, rowLift, operands[1 - broadcastIndex], vec, rewriter)->getResults());
      return true;
    }
    return false;
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
  /// The reshaped extent names the source tile's allocated extent —
  /// `pl.reshape` conserves physical elements, so the view must match
  /// the padded bound the source's producing read or reduction
  /// allocated, never the broadcast's aligned max size (a valid bound);
  /// the max size stands in when the allocation is not statically
  /// resolvable. A static source extent is a valid bound too: PTOAS
  /// allocates every tile at the aligned bound, so the view still
  /// carries the allocated extent, spelled as a constant the emitpy
  /// lowering renders literally. The target type is static when all
  /// extents are known and dynamic only for runtime allocation bounds.
  static FailureOr<std::pair<Type, SmallVector<Value>>> buildLiftedVector(npuvector::BroadcastOp op,
                                                                          npuvector::NPUVectorType srcType,
                                                                          bool rowLift,
                                                                          ConversionPatternRewriter &rewriter) {
    Location loc = op.getLoc();
    // The mapped source already names its physical allocation. Reuse that
    // shape directly, without inferring padding again from the source IR.
    if (auto source = dyn_cast<pypto::TileType>(mapped(rewriter, op.getSource()).getType());
        source && source.getRank() == 1 && !ShapedType::isDynamic(source.getShape()[0])) {
      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value extent = rewriter.create<arith::ConstantIndexOp>(loc, source.getShape()[0]);
      Type vecType = srcType.clone(rowLift ? SmallVector<int64_t>{ShapedType::kDynamic, 1}
                                          : SmallVector<int64_t>{1, ShapedType::kDynamic});
      return std::make_pair(vecType, rowLift ? SmallVector<Value>{extent, one} : SmallVector<Value>{one, extent});
    }
    int64_t srcExtent = srcType.getDimSize(0);
    if (ShapedType::isDynamic(srcExtent)) {
      ValueRange maxSizes = op.getMaxSizes();
      if (maxSizes.size() != 2) {
        return failure();
      }
      Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value extent;
      if (auto shape = tileAllocationShape(op.getSource()); shape && shape->size() == 1) {
        extent = rewriter.create<arith::ConstantIndexOp>(loc, (*shape)[0]);
      } else {
        // Unresolvable allocation: pad a constant max size to whole
        // 32-byte rows, matching the aligned extent a dynamic
        // transfer_read allocates, so the reshape stays
        // shape-compatible with the source tile (42 -> 48 for FP32).
        // Mirrors `buildBroadcastExtents`.
        Value max = rowLift ? maxSizes[0] : maxSizes[1];
        if (auto bound = constIndexValue(max)) {
          unsigned elemBytes = srcType.getElementType().getIntOrFloatBitWidth() / 8;
          extent = rewriter.create<arith::ConstantIndexOp>(loc, alignExtentTo32B(*bound, elemBytes));
        } else {
          extent = max;
        }
      }
      if (rowLift) {
        Type vecType = srcType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
        return std::make_pair(vecType, SmallVector<Value>{extent, one});
      }
      Type vecType = srcType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
      return std::make_pair(vecType, SmallVector<Value>{one, extent});
    }
    // Static source extent: the allocated bound the view must match.
    unsigned elemBytes = srcType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t bound;
    if (auto shape = tileAllocationShape(op.getSource()); shape && shape->size() == 1) {
      bound = (*shape)[0];
    } else {
      bound = alignExtentTo32B(srcExtent, elemBytes);
    }
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value extent = rewriter.create<arith::ConstantIndexOp>(loc, bound);
    if (rowLift) {
      Type vecType = srcType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
      return std::make_pair(vecType, SmallVector<Value>{extent, one});
    }
    Type vecType = srcType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
    return std::make_pair(vecType, SmallVector<Value>{one, extent});
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
    tile = mapped(rewriter, tile);
    vec = mapped(rewriter, vec);
    auto tileType = cast<pypto::TileType>(tile.getType());
    Type result = pyptoType(user->getResult(0).getType(), tileType.getShape());
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

static bool fuseTileBroadcastConsumer(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter) {
  return TileBroadcastFusionPattern::tryFuseConsumer(op, operands, rewriter);
}

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
  /// The reshaped vector names the source tile's allocated extent --
  /// `pl.reshape` conserves physical elements, so the view must match
  /// the padded bound the source's producing read or reduction
  /// allocated. The target Tile type records those physical extents,
  /// independently of the valid region inherited from the source.
  static Value liftOntoCarrier(Location loc, npuvector::NPUVectorType valueTileType, Value carrier, bool rowLift,
                               Value tile, ConversionPatternRewriter &rewriter) {
    auto tileType = cast<ShapedType>(tile.getType());
    unsigned elemBytes = tileType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t bound = tileType.getDimSize(0);
    if (auto shape = tileAllocationShape(tile); shape && shape->size() == 1 && !ShapedType::isDynamic((*shape)[0])) {
      bound = (*shape)[0];
    } else {
      bound = alignExtentTo32B(bound, elemBytes);
    }
    Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
    Value extent = rewriter.create<arith::ConstantIndexOp>(loc, bound);
    Type vecType;
    SmallVector<Value> vecExtents;
    if (rowLift) {
      vecType = tileType.clone(SmallVector<int64_t>{ShapedType::kDynamic, 1});
      vecExtents.push_back(extent);
      vecExtents.push_back(one);
    } else {
      vecType = tileType.clone(SmallVector<int64_t>{1, ShapedType::kDynamic});
      vecExtents.push_back(one);
      vecExtents.push_back(extent);
    }
    Value vec =
      rewriter.create<pypto::ReshapeOp>(loc, pyptoExtentType(vecType, ValueRange(vecExtents)), mapped(rewriter, tile),
                                        mapped(rewriter, ValueRange(vecExtents)), ValueRange());
    // The carrier type names the padded allocation; its reshape keeps the
    // smaller logical valid region independently of this result type.
    Type resultType = pyptoType(valueTileType, cast<pypto::TileType>(carrier.getType()).getShape());
    if (rowLift) {
      return rewriter.create<pypto::RowExpandAddOp>(loc, resultType, carrier, mapped(rewriter, vec))
        .getResult();
    }
    return rewriter.create<pypto::ColExpandAddOp>(loc, resultType, carrier, mapped(rewriter, vec))
      .getResult();
  }

  /// The aligned bound each result axis allocates (the comparison tile's
  /// allocated extent on the lifted axis) raised onto the sibling grid:
  /// PTOAS lays every tile in the block onto one grid, so the zero
  /// carrier must cover the allocation that grid gives the mask's
  /// consumers — the allocated extents of the value tiles the guarding
  /// select combines — never the valid extents the result type carries,
  /// or the expand below mismatches its [1, 32] view against a [2, 4]
  /// carrier and `pl.tile.select` mismatches the tiles it guards.
  static SmallVector<Value> cmpCarrierExtents(ShapedType resultType, npuvector::NPUVectorType lhsType, Value cmpLhs,
                                              bool rowLift, Location loc, const SmallVectorImpl<ReductionAxisPad> &pads,
                                              const SmallVectorImpl<int64_t> &siblingBounds,
                                              ConversionPatternRewriter &rewriter) {
    unsigned rank = resultType.getRank();
    unsigned identityAxis = rowLift ? 0 : 1;
    unsigned elemBytes = lhsType.getElementType().getIntOrFloatBitWidth() / 8;
    SmallVector<Value> carrierExtents;
    for (unsigned axis = 0; axis < rank; ++axis) {
      int64_t bound;
      if (axis == identityAxis) {
        // The lifted axis: the comparison tile's allocated extent,
        // falling back to an aligned bound, so the carrier matches the
        // reshaped vector the expand combines with it.
        bound = lhsType.getDimSize(0);
        if (auto shape = tileAllocationShape(cmpLhs); shape && shape->size() == 1 &&
                                                        !ShapedType::isDynamic((*shape)[0])) {
          bound = (*shape)[0];
        } else {
          bound = alignExtentTo32B(bound, elemBytes);
        }
      } else {
        bound = padAllocationExtent(resultType.getDimSize(axis), axis, rank, elemBytes, pads);
      }
      if (axis < siblingBounds.size() && !ShapedType::isDynamic(siblingBounds[axis])) {
        bound = std::max(bound, siblingBounds[axis]);
      }
      carrierExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, bound));
    }
    return carrierExtents;
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
    // PTOAS lays every tile in the block onto one grid, so the zero
    // carrier must cover the allocation that grid gives the mask's
    // consumers: the aligned bound each result axis allocates (the
    // comparison tile's allocated extent on the lifted axis) and the
    // allocated extents of the value tiles the guarding select
    // combines -- never the valid extents the result type carries, or
    // the expand below mismatches its [1, 32] view against a [2, 4]
    // carrier and `pl.tile.select` mismatches the tiles it guards.
    unsigned rank = resultType.getRank();
    unsigned elemBytes = lhsType.getElementType().getIntOrFloatBitWidth() / 8;
    SmallVector<ReductionAxisPad> pads;
    collectReductionAxisPads(op->getResult(0), 0, pads);
    SmallVector<int64_t> siblingBounds;
    collectBroadcastSiblingBounds(op, rank, siblingBounds);
    SmallVector<Value> carrierExtents =
      cmpCarrierExtents(resultType, lhsType, cmpLhs, rowLift, loc, pads, siblingBounds, rewriter);
    Value carrier =
      rewriter
        .create<pypto::TileFullOp>(loc, pyptoExtentType(valueTileType, ValueRange(carrierExtents)),
                                   rewriter.getStringAttr(zero), mapped(rewriter, ValueRange(carrierExtents)))
        .getResult();
    // The carrier allocates the padded sibling shape while valueTileType
    // names the valid contract; re-view it back down to that valid region:
    // the expand preserves the target tile's valid shape, tile.cmps packs
    // the source's valid extent into the predicate mask, and tile.select
    // checks that mask against the valid extents of the value branches it
    // guards -- a carrier left at the padded valid extent would pack
    // ceil(32/8) mask columns where the select expects ceil(valid/8).
    SmallVector<Value> carrierValid;
    std::transform(resultType.getShape().begin(), resultType.getShape().end(), std::back_inserter(carrierValid),
                   [&](int64_t dim) { return rewriter.create<arith::ConstantIndexOp>(loc, dim); });
    carrier = rewriter
                .create<pypto::ReshapeOp>(loc, pyptoExtentType(valueTileType, ValueRange(carrierExtents)), carrier,
                                          mapped(rewriter, ValueRange(carrierExtents)),
                                          mapped(rewriter, ValueRange(carrierValid)))
                .getResult();
    Value liftedLhs = liftOntoCarrier(loc, valueTileType, carrier, rowLift, cmpLhs, rewriter);
    // A scalar rhs re-rides the tile-scalar comparison; a tile rhs lifts.
    // A rhs that merely broadcasts a scalar passes through to that
    // scalar: lifting the broadcast result would hang a reshape off an
    // op the scalar-broadcast fusion erases, whichever pattern runs
    // first.
    Value finalRhs = cmpRhs;
    if (auto rhsBroadcast = cmpRhs.getDefiningOp<npuvector::BroadcastOp>()) {
      if (!isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(rhsBroadcast.getSource().getType())) {
        finalRhs = rhsBroadcast.getSource();
      }
    }
    if (isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(finalRhs.getType())) {
      finalRhs = liftOntoCarrier(loc, valueTileType, carrier, rowLift, finalRhs, rewriter);
    }
    Type maskType = pyptoType(op->getResult(0).getType(), cast<pypto::TileType>(liftedLhs.getType()).getShape());
    Value mask = rewriter.create<pypto::CmpOp>(loc, maskType, *predicate,
                                               mapped(rewriter, liftedLhs), mapped(rewriter, finalRhs));
    replaceWithPackedMask(op, mask, rewriter);
    return success();
  }
};

/// Fuses a scalar broadcast into every fusable consumer: `cmp pred,
/// %tile, broadcast(%scalar)` becomes `pypto.cmp pred, %tile, %scalar`
/// (lowered onto the tile-scalar `pl.tile.cmps`), and
/// `select(%cond, broadcast(%scalar), %tile)` becomes `pypto.select
/// (%cond, %scalar, %tile)` (the composite `pl.tile.select` accepts a
/// literal scalar branch and synthesizes its own scratch). Float
/// division with the broadcast on its lhs becomes
/// `pypto.mul(pypto.recip %tile, %scalar)` — division is the one binary
/// whose materialized lhs would still fail `pl.div`'s same-valid-shape
/// contract — and subtraction with the broadcast on its lhs becomes
/// `pypto.add(pypto.neg %tile, %scalar)`, keeping the tile-first operand
/// order the runtime's `pl.sub` demands. A materialized broadcast tile
/// cannot follow the loads'
/// shape convention (leading 1s to the source rank plus 32B-aligned
/// columns — the alignment is dtype-relative, so an f32 carrier drifts
/// from a bf16-allocated load cast into f32), so the scalar forms
/// sidestep tile-shape matching entirely.
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
    return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
               arith::DivSIOp, arith::DivUIOp>(op);
  }

  /// Commutative binaries may take the broadcast on either side; the
  /// operands swap into the runtime's tile-first form.
  static bool isCommutativeArith(Operation *op) {
    return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(op);
  }

  /// Float min/max clamp bounds fuse onto the runtime's tile-scalar
  /// `maximums`/`minimums` forms (TMAXS/TMINS): the clamp is commutative,
  /// so the broadcast may sit on either side.
  static bool isFusableClampBinary(Operation *op) {
    return isa<arith::MaximumFOp, arith::MinimumFOp>(op);
  }

  /// A non-finite clamp bound (-inf/+inf/nan) must stay on the `tile_full`
  /// materialization: the runtime spells a tile-full constant with its hex
  /// bit pattern (`0xFF800000`), which ptoas parses, while a scalar
  /// `pl.const` prints the bare `-inf` literal, which ptoas rejects. Both
  /// constant spellings are probed, mirroring `isLiteralScalar`.
  static bool isNonFiniteScalarConstant(Value scalar) {
    if (auto pyptoConst = scalar.getDefiningOp<pypto::ConstantOp>()) {
      // floatToPythonText renders non-finite values as the lossless
      // `pl._dump.float64('...')` spelling.
      return pyptoConst.getValue().starts_with("pl._dump.");
    }
    auto arithConst = scalar.getDefiningOp<arith::ConstantOp>();
    if (!arithConst) {
      return false;
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(arithConst.getValue())) {
      return !floatAttr.getValue().isFinite();
    }
    return false;
  }

  /// Match gate for a binary arith user: the broadcast must sit on the
  /// rhs of a tile operand, or on the lhs where the op still commutes
  /// (the rewrite moves the scalar to the rhs) or where the rewrite can
  /// spell it from the tile alone — float division through
  /// `muls(recip(tile), scalar)`, subtraction through `adds(neg(tile),
  /// scalar)`.
  static bool isFusibleArithBinaryUser(Operation *user, Value bcast, bool isTileType(Type)) {
    Value lhs = user->getOperand(0);
    Value rhs = user->getOperand(1);
    if (rhs == bcast) {
      return isTileType(lhs.getType());
    }
    if (lhs == bcast && isCommutativeArith(user)) {
      return isTileType(rhs.getType());
    }
    // Float division also fuses with the broadcast on its lhs: the
    // rewrite spells `scalar / tile` as `muls(recip(tile), scalar)`.
    if (lhs == bcast && isa<arith::DivFOp>(user)) {
      return isTileType(rhs.getType());
    }
    // Subtraction likewise fuses with the broadcast on its lhs: the
    // rewrite spells `scalar - tile` as `adds(neg(tile), scalar)`.
    if (lhs == bcast && isa<arith::SubFOp, arith::SubIOp>(user)) {
      return isTileType(rhs.getType());
    }
    return false;
  }

  /// Match gate for a clamp-style binary user: the broadcast must sit on
  /// either side of a tile operand, and its scalar source must be finite —
  /// non-finite bounds (softmax's -inf reduction seed) retreat to the
  /// tile_full materialization — see isNonFiniteScalarConstant.
  static bool isFusibleClampBinaryUser(Operation *user, Value bcast, bool isTileType(Type)) {
    Value lhs = user->getOperand(0);
    Value rhs = user->getOperand(1);
    if (lhs != bcast && rhs != bcast) {
      return false;
    }
    Value other = lhs == bcast ? rhs : lhs;
    if (!isTileType(other.getType())) {
      return false;
    }
    Value scalar = cast<npuvector::BroadcastOp>(bcast.getDefiningOp()).getSource();
    return !isNonFiniteScalarConstant(scalar);
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
      return isFusibleArithBinaryUser(user, bcast, isTileType);
    }
    if (isFusableClampBinary(user)) {
      return isFusibleClampBinaryUser(user, bcast, isTileType);
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

  /// Rewrites one matched consumer onto its scalar form in place: the
  /// comparison keeps its tile lhs with the scalar rhs (packed through
  /// the mask rewriter), binary arith swaps into the runtime's
  /// tile-first form, clamps map onto the tile-scalar TMAXS/TMINS forms,
  /// and selects take the scalar on the broadcast's branch.
  static void rewriteFusedUser(Operation *user, Value scalar, Value bcast, ConversionPatternRewriter &rewriter,
                               ValueRange operands = {}) {
    if (auto cmp = dyn_cast<npuvector::CmpIOp>(user)) {
      auto code = cmpTypeCodeFor(cmp.getPredicate());
      SmallVector<Value> mappedOperands{mapped(rewriter, cmp.getLhs()), mapped(rewriter, scalar)};
      Value mask = rewriter.create<pypto::CmpOp>(cmp.getLoc(), pyptoResultTypes(cmp, mappedOperands).front(),
                                                 static_cast<pypto::CmpPredicate>(*code),
                                                 mappedOperands[0], mappedOperands[1]);
      replaceWithPackedMask(cmp, mask, rewriter);
    } else if (auto cmp = dyn_cast<npuvector::CmpFOp>(user)) {
      auto code = cmpTypeCodeFor(cmp.getPredicate());
      SmallVector<Value> mappedOperands{mapped(rewriter, cmp.getLhs()), mapped(rewriter, scalar)};
      Value mask = rewriter.create<pypto::CmpOp>(cmp.getLoc(), pyptoResultTypes(cmp, mappedOperands).front(),
                                                 static_cast<pypto::CmpPredicate>(*code),
                                                 mappedOperands[0], mappedOperands[1]);
      replaceWithPackedMask(cmp, mask, rewriter);
    } else if (isFusableArithBinary(user)) {
      // `tile OP scalar`: the broadcast sits on the rhs, or (commutative
      // ops) on the lhs with the operands swapped into the runtime's
      // tile-first form.
      Value tile = user->getOperand(1) == bcast ? user->getOperand(0) : user->getOperand(1);
      if (isa<arith::AddFOp, arith::AddIOp>(user)) {
        rewriter.replaceOpWithNewOp<pypto::AddOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                  mapped(rewriter, scalar));
      } else if (isa<arith::SubFOp, arith::SubIOp>(user)) {
        if (user->getOperand(0) == bcast) {
          // `scalar - tile`: the runtime's `pl.sub` is tile-first, so
          // spell the difference as `adds(neg(tile), scalar)`. The
          // negation preserves the tile's own shape and the
          // tile-scalar add keeps it, so the pair needs no shape
          // contract on the materialized broadcast.
          SmallVector<Type> resultTypes = pyptoResultTypes(user, operands);
          Value neg = rewriter.create<pypto::NegOp>(user->getLoc(), resultTypes, mapped(rewriter, tile));
          rewriter.replaceOpWithNewOp<pypto::AddOp>(user, resultTypes, neg, mapped(rewriter, scalar));
        } else {
          rewriter.replaceOpWithNewOp<pypto::SubOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                    mapped(rewriter, scalar));
        }
      } else if (isa<arith::MulFOp, arith::MulIOp>(user)) {
        rewriter.replaceOpWithNewOp<pypto::MulOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                  mapped(rewriter, scalar));
      } else if (isa<arith::DivFOp>(user) && user->getOperand(0) == bcast) {
        // `scalar / tile`: `pl.div` demands src0/src1/dst share one
        // valid shape, but a materialized broadcast carries the
        // broadcast's static bound, never the tile's dynamic valid
        // extent. `pl.recip` preserves the tile's own shape and the
        // tile-scalar mul keeps it, so the pair spells the quotient
        // without any shape contract.
        SmallVector<Type> resultTypes = pyptoResultTypes(user, operands);
        Value recip = rewriter.create<pypto::RecipOp>(user->getLoc(), resultTypes, mapped(rewriter, tile));
        rewriter.replaceOpWithNewOp<pypto::MulOp>(user, resultTypes, recip, mapped(rewriter, scalar));
      } else {
        rewriter.replaceOpWithNewOp<pypto::DivOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                  mapped(rewriter, scalar));
      }
    } else if (isFusableClampBinary(user)) {
      // `min/max(tile, scalar)`: the runtime's TMAXS/TMINS take the
      // scalar directly, so no `tile_full` carrier (and no Vec buffer
      // allocation for it) is materialized. The clamp is commutative,
      // so the broadcast may sit on either side.
      Value tile = user->getOperand(1) == bcast ? user->getOperand(0) : user->getOperand(1);
      if (isa<arith::MaximumFOp>(user)) {
        rewriter.replaceOpWithNewOp<pypto::MaximumsOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                       mapped(rewriter, scalar));
      } else {
        rewriter.replaceOpWithNewOp<pypto::MinimumsOp>(user, pyptoResultTypes(user, operands), mapped(rewriter, tile),
                                                       mapped(rewriter, scalar));
      }
    } else {
      auto select = cast<npuvector::SelectOp>(user);
      bool trueIsBcast = select.getTrueValue() == bcast;
      SmallVector<Value> mappedOperands{mapped(rewriter, trueIsBcast ? select.getFalseValue() : select.getTrueValue()),
                                          mapped(rewriter, scalar)};
      rewriter.replaceOpWithNewOp<pypto::SelectOp>(select, pyptoResultTypes(select, mappedOperands).front(),
                                                   mapped(rewriter, select.getCondition()),
                                                   mapped(rewriter, trueIsBcast ? scalar : select.getTrueValue()),
                                                   mapped(rewriter, trueIsBcast ? select.getFalseValue() : scalar));
    }
  }

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, npuvector::BroadcastOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Both source vector forms and already-converted Tiles can feed fusion.
    auto isTileType = [](Type type) { return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(type); };
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
    for (Operation *user : users) {
      for (Value operand : user->getOperands()) {
        if (operand == op.getResult() || llvm::is_contained(users, operand.getDefiningOp())) {
          continue;
        }
        if (isa<npuvector::NPUVectorType>(mapped(rewriter, operand).getType())) {
          return rewriter.notifyMatchFailure(op, "fusion data producer has not been converted yet");
        }
      }
    }
    // A select may consume a comparison that uses the same broadcast.
    // Rewrite that producer first so the select receives its mapped Tile.
    if (!computeTopologicalSorting(users)) {
      return rewriter.notifyMatchFailure(op, "broadcast consumers have cyclic dependencies");
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
      rewriteFusedUser(user, scalar, op.getResult(), rewriter);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Every comparison/select consumer must accept the same scalar fusion.
static bool scalarBroadcastUsersAreFusible(npuvector::BroadcastOp broadcast) {
  auto isTileType = [](Type type) { return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(type); };
  if (llvm::any_of(broadcast->getUsers(), [&](Operation *user) {
        return !ScalarBroadcastFusionPattern::isFusibleUser(user, broadcast.getResult(), isTileType);
      })) {
    return false;
  }
  return !llvm::any_of(broadcast->getUsers(), [](Operation *user) { return isa<npuvector::SelectOp>(user); }) ||
         ScalarBroadcastFusionPattern::isLiteralScalar(broadcast.getSource());
}

static bool hasScalarBroadcastConsumerForm(Operation *op, size_t operandCount) {
  if (isa<npuvector::SelectOp>(op)) {
    return operandCount == 3;
  }
  return operandCount == 2 &&
         (isa<npuvector::CmpIOp, npuvector::CmpFOp>(op) || ScalarBroadcastFusionPattern::isFusableArithBinary(op) ||
          ScalarBroadcastFusionPattern::isFusableClampBinary(op));
}

/// A broadcast may precede the data producer, so its producer-side fusion
/// cannot yet construct the converted Tile. Reuse the same scalar rewrite
/// at the consumer once its mapped data Tile is available.
static bool fuseScalarBroadcastConsumer(Operation *op, ValueRange operands, ConversionPatternRewriter &rewriter) {
  bool select = isa<npuvector::SelectOp>(op);
  bool comparison = isa<npuvector::CmpIOp, npuvector::CmpFOp>(op);
  if (!hasScalarBroadcastConsumerForm(op, operands.size())) {
    return false;
  }
  auto isTileType = [](Type type) { return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(type); };
  SmallVector<unsigned> scalarIndices = select ? SmallVector<unsigned>{1u, 2u} : SmallVector<unsigned>{1u, 0u};
  for (unsigned scalarIndex : scalarIndices) {
    auto broadcast = op->getOperand(scalarIndex).getDefiningOp<npuvector::BroadcastOp>();
    unsigned tileIndex = select ? 3 - scalarIndex : 1 - scalarIndex;
    if (!broadcast || isTileType(broadcast.getSource().getType()) ||
        !isa<pypto::TileType>(operands[tileIndex].getType()) ||
        (select && (!isa<pypto::TileType>(operands[0].getType()) ||
                    !ScalarBroadcastFusionPattern::isLiteralScalar(broadcast.getSource()))) ||
        !ScalarBroadcastFusionPattern::isFusibleUser(op, broadcast.getResult(), isTileType)) {
      continue;
    }
    if ((select || comparison) && !scalarBroadcastUsersAreFusible(broadcast)) {
      continue;
    }
    SmallVector<Value> fusedOperands{operands[tileIndex], mapped(rewriter, broadcast.getSource())};
    ScalarBroadcastFusionPattern::rewriteFusedUser(op, broadcast.getSource(), broadcast.getResult(), rewriter,
                                                  fusedOperands);
    return true;
  }
  return false;
}

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
      rewriter.replaceOp(op, adaptor.getVector());
      return success();
    }
    Type resultType = pyptoType(op->getResult(0).getType());
    if (isa<npuvector::NPUVectorType>(op->getResult(0).getType())) {
      auto source = cast<ShapedType>(adaptor.getVector().getType());
      SmallVector<int64_t> shape(source.getShape());
      std::swap(shape[swap->first], shape[swap->second]);
      resultType = pyptoType(resultType, shape);
    }
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(
      op, resultType, adaptor.getVector(), static_cast<uint64_t>(swap->first), static_cast<uint64_t>(swap->second));
    return success();
  }
};

/// Materializes a select branch for `pypto.select`: tile branches pass
/// through unchanged, while constant scalar branches become full tiles --
/// that materialization needs a static shape, since a dynamic tile_full
/// would have no extents to bind. Returns a null value when the branch is
/// neither a tile nor a materializable constant.
static Value materializeSelectBranch(Value branch, bool staticShape, Type convertedResult, Location loc,
                                     ConversionPatternRewriter &rewriter) {
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
  return rewriter
    .create<pypto::TileFullOp>(loc, pyptoExtentType(convertedResult, ValueRange{}), rewriter.getStringAttr(*text),
                               ValueRange{})
    .getResult();
}

/// Re-spells a constant select branch full with 32-byte-padded allocation
/// rows: the full spells the padded allocation in its extents while its
/// result type keeps naming the valid shape, the same split the
/// sibling-bounds raise on `tile_full` uses; the reshape keeps that static
/// type too -- the verifiers only check the element count against the
/// full's own type and the select demands the branches match its result
/// shape verbatim.
static Value padSelectBranchFull(pypto::TileFullOp full, int64_t paddedCols, ArrayRef<int64_t> validShape,
                                 Type convertedResult, Location loc, ConversionPatternRewriter &rewriter) {
  SmallVector<Value> extents;
  SmallVector<Value> validExtents;
  for (unsigned axis = 0; axis < validShape.size(); ++axis) {
    int64_t allocation = axis + 1 == validShape.size() ? paddedCols : validShape[axis];
    extents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, allocation));
    validExtents.push_back(rewriter.create<arith::ConstantIndexOp>(loc, validShape[axis]));
  }
  auto paddedFull = rewriter.create<pypto::TileFullOp>(loc, pyptoExtentType(convertedResult, ValueRange(extents)),
                                                       full.getValue(), mapped(rewriter, ValueRange(extents)));
  return rewriter.create<pypto::ReshapeOp>(loc, pyptoExtentType(convertedResult, ValueRange(extents)), paddedFull,
                                           mapped(rewriter, ValueRange(extents)),
                                           mapped(rewriter, ValueRange(validExtents)));
}

/// Rewrites a select into `pypto.select`: the runtime's composite
/// `pl.tile.select` synthesizes the architecture-defined scratch tile, so
/// no scratch is materialized here and dynamic result shapes are legal.
/// Constant scalar branches are materialized as `pypto.tile_full` (which
/// needs static shapes, since the extents would otherwise be unknown);
/// tile branches use their mapped target values. Scalar selects (e.g. the
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
    if (fuseScalarBroadcastConsumer(op, adaptor.getOperands(), rewriter)) {
      return success();
    }

    Location loc = op.getLoc();
    Type convertedResult = pyptoResultType(op->getResult(0));
    // Converted data branches carry the physical shape; the source graph
    // may still describe a smaller logical allocation.
    for (Value branch : adaptor.getOperands().drop_front()) {
      if (auto tile = dyn_cast<pypto::TileType>(branch.getType())) {
        convertedResult = pyptoType(convertedResult, tile.getShape());
        break;
      }
    }
    bool staticShape = resultType.hasStaticShape();

    Value trueBranch = materializeSelectBranch(op.getTrueValue(), staticShape, convertedResult, loc, rewriter);
    Value falseBranch = materializeSelectBranch(op.getFalseValue(), staticShape, convertedResult, loc, rewriter);
    if (!trueBranch || !falseBranch) {
      return rewriter.notifyMatchFailure(op,
                                         "select branches must be tiles, or constant scalars on a "
                                         "statically shaped select");
    }

    // Both branches materialized as full tiles: the full's own dtype may
    // already lay 32-byte-aligned rows that a downstream cast narrows
    // below the alignment -- a cast clones the source tile's allocation
    // (`DeduceTileCastType`), so an INT32 select over [2704] whose result
    // feeds a FP16-then-INT8 chain makes the INT8 tile inherit 2704-byte
    // rows, which `pto.alloc_tile` rejects. Pad the fulls' allocation to
    // the narrowest downstream dtype and trim the valid region back to
    // the source shape. Both branches take the same treatment so the
    // frontend's shape/valid/dtype equality still holds, and the emitpy
    // lowering turns the valid-carrying reshape into the reshape+slice
    // sandwich the runtime accepts. A mixed branch (one tile value, one
    // constant) keeps the unpadded form: its geometry must match the
    // tile branch, which this pad cannot know.
    auto trueFull = trueBranch.getDefiningOp<pypto::TileFullOp>();
    auto falseFull = falseBranch.getDefiningOp<pypto::TileFullOp>();
    if (trueFull && falseFull) {
      int64_t cols = resultType.getShape().back();
      auto minBytes = castChainMinElemBytes(op->getResult(0));
      if (minBytes && cols > 0 && (cols * static_cast<int64_t>(*minBytes)) % 32 != 0) {
        int64_t paddedCols = alignExtentTo32B(cols, *minBytes);
        ArrayRef<int64_t> validShape = resultType.getShape();
        trueBranch = padSelectBranchFull(trueFull, paddedCols, validShape, convertedResult, loc, rewriter);
        falseBranch = padSelectBranchFull(falseFull, paddedCols, validShape, convertedResult, loc, rewriter);
        rewriter.eraseOp(trueFull);
        rewriter.eraseOp(falseFull);
      }
    }

    // The target result carries the physical allocation, while branch
    // views retain the valid region for the positional select call.
    rewriter.replaceOpWithNewOp<pypto::SelectOp>(op, pyptoType(convertedResult), mapped(rewriter, op.getCondition()),
                                                 mapped(rewriter, trueBranch), mapped(rewriter, falseBranch));
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
    // This is a Tensor view, so its memref result remains until the
    // downstream pass emits the Tensor annotation.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, pyptoExtentType(op.getResult().getType(), ValueRange()),
                                                  mapped(rewriter, op.getSrc()), ValueRange(), ValueRange());
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
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, pyptoExtentType(op.getResult().getType(), ValueRange()),
                                                  mapped(rewriter, op.getSrc()), ValueRange(), ValueRange());
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
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(op, pyptoType(op.getResult().getType()),
                                                    mapped(rewriter, op.getIn()), static_cast<uint64_t>(swap->first),
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
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, pyptoExtentType(op.getResult().getType(), ValueRange()),
                                                  mapped(rewriter, op.getSource()), ValueRange(), ValueRange());
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
    rewriter.replaceOpWithNewOp<pypto::SliceOp>(op, pyptoResultType(op.getResult()), adaptor.getSource(), window,
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
    auto loop = rewriter.create<pypto::RangeOp>(op.getLoc(), mapped(rewriter, op.getLowerBound()),
                                                mapped(rewriter, op.getUpperBound()), mapped(rewriter, op.getStep()));
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

/// One streaming-reduction tail window matched for the
/// insert-slice-into-reduction peephole: the reduction over the inserted
/// window, the insert itself, the window combine, the tail operand, and
/// the window extract.
struct InsertSliceFoldCandidate {
  npuvector::ReductionOp red;
  npuvector::InsertSliceOp ins;
  Operation *bin;
  Value tail;
  npuvector::ExtractSliceOp ext;
};

/// The fold splits the reduction across the insert window; add, max and
/// min all obey the absorption law because the extracted window carries
/// every scratch value the insert overwrites.
static bool isAbsorbableReductionKind(vector::CombiningKind kind) {
  return kind == vector::CombiningKind::ADD || kind == vector::CombiningKind::MAXIMUMF ||
         kind == vector::CombiningKind::MINIMUMF || kind == vector::CombiningKind::MAXSI ||
         kind == vector::CombiningKind::MAXUI || kind == vector::CombiningKind::MINSI ||
         kind == vector::CombiningKind::MINUI;
}

/// Only full reductions qualify: the fold sums the whole scratch tile, so
/// every source axis must be reduced.
static bool isFullReduction(npuvector::ReductionOp red, ShapedType shaped) {
  if (auto dims = red.getReductionDims(); dims && !dims->empty()) {
    if (dims->size() != static_cast<size_t>(shaped.getRank())) {
      return false;
    }
    llvm::SmallDenseSet<int64_t> covered;
    for (int64_t d : *dims) {
      if (d < 0 || d >= static_cast<int64_t>(shaped.getRank()) || !covered.insert(d).second) {
        return false;
      }
    }
  }
  return true;
}

/// The window combine must mirror the reduction kind: add pairs with
/// addf/addi, max/maxui/maxsi with the matching arith spelling, and so
/// does the min family.
static bool binMatchesReductionKind(vector::CombiningKind kind, Operation *bin) {
  return (kind == vector::CombiningKind::ADD && isa<arith::AddFOp, arith::AddIOp>(bin)) ||
         (kind == vector::CombiningKind::MAXIMUMF && isa<arith::MaximumFOp>(bin)) ||
         (kind == vector::CombiningKind::MINIMUMF && isa<arith::MinimumFOp>(bin)) ||
         (kind == vector::CombiningKind::MAXSI && isa<arith::MaxSIOp>(bin)) ||
         (kind == vector::CombiningKind::MAXUI && isa<arith::MaxUIOp>(bin)) ||
         (kind == vector::CombiningKind::MINSI && isa<arith::MinSIOp>(bin)) ||
         (kind == vector::CombiningKind::MINUI && isa<arith::MinUIOp>(bin));
}

static bool sameValueRanges(ValueRange a, ValueRange b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto [x, y] : llvm::zip(a, b)) {
    if (x != y) {
      return false;
    }
  }
  return true;
}

/// The bin combines the tail increment with the window extracted back
/// out of the scratch tile: exactly one operand must be that extract,
/// the other is the tail whose sum the fold splits out. Fails on two
/// window extracts (ambiguous) or two plain operands.
static bool disambiguateWindowOperands(Operation *bin, Value &tail, npuvector::ExtractSliceOp &ext) {
  for (Value operand : bin->getOperands()) {
    if (auto candidate = operand.getDefiningOp<npuvector::ExtractSliceOp>()) {
      if (ext) {
        return false;  // two window extracts: ambiguous
      }
      ext = candidate;
    } else if (tail) {
      return false;  // two non-extract operands
    } else {
      tail = operand;
    }
  }
  return ext != nullptr;
}

/// The extract must read the same window the insert writes: the same dest
/// buffer and identical offset/size/stride packs. A rank-reduced extract
/// window changes the covered-position set; only full-rank windows (one
/// keep_dim per source axis) keep the sum linear.
static bool extractMatchesInsertWindow(npuvector::InsertSliceOp ins, npuvector::ExtractSliceOp ext) {
  if (ext.getSource() != ins.getDest()) {
    return false;
  }
  if (!sameValueRanges(ins.getOffsets(), ext.getOffsets()) || !sameValueRanges(ins.getSizes(), ext.getSizes()) ||
      !sameValueRanges(ins.getStrides(), ext.getStrides())) {
    return false;
  }
  return ext.getKeepDims().size() == ins.getOffsets().size();
}

/// Matches one reduction against the foldable streaming-tail shape:
/// `reduction(insert_slice(bin(tail, extract_slice(scratch, win)),
/// scratch, win))`. Returns nullopt when any leg of that shape is absent.
static std::optional<InsertSliceFoldCandidate> matchInsertSliceFold(npuvector::ReductionOp red) {
  vector::CombiningKind kind = red.getKind();
  if (!isAbsorbableReductionKind(kind) || red.getAcc()) {
    return std::nullopt;
  }
  auto shaped = dyn_cast<ShapedType>(red.getVector().getType());
  if (!shaped || !isFullReduction(red, shaped)) {
    return std::nullopt;
  }
  auto ins = red.getVector().getDefiningOp<npuvector::InsertSliceOp>();
  if (!ins) {
    return std::nullopt;
  }
  Operation *bin = ins.getSource().getDefiningOp();
  if (!binMatchesReductionKind(kind, bin) || bin->getNumOperands() != 2) {
    return std::nullopt;
  }
  InsertSliceFoldCandidate candidate;
  if (!disambiguateWindowOperands(bin, candidate.tail, candidate.ext) ||
      !extractMatchesInsertWindow(ins, candidate.ext)) {
    return std::nullopt;
  }
  candidate.red = red;
  candidate.ins = ins;
  candidate.bin = bin;
  return candidate;
}

/// Replaces one matched candidate: the reduction over the inserted window
/// becomes the combine of the scratch and tail reductions, and the folded
/// insert/extract/bin ops drop away when nothing else uses them.
static void applyInsertSliceFold(InsertSliceFoldCandidate &item) {
  OpBuilder builder(item.red);
  Location loc = item.red.getLoc();
  Type resultType = item.red->getResult(0).getType();
  auto cloneOver = [&](Value vector) -> Value {
    Operation *clone = builder.clone(*item.red);
    clone->setOperand(0, vector);
    return clone->getResult(0);
  };
  Value scratchSum = cloneOver(item.ins.getDest());
  Value tailSum = cloneOver(item.tail);
  Value sum;
  if (isa<arith::AddFOp, arith::AddIOp>(item.bin)) {
    // Add recombines with the source op itself: clone the bin, land the
    // scalar result type, and drop the vectorizer's reduction markers so
    // the scalar combine cannot be misattributed as a partial-reduction
    // site.
    Operation *sumOp = builder.clone(*item.bin);
    sumOp->setOperands(ValueRange({scratchSum, tailSum}));
    sumOp->getResult(0).setType(resultType);
    sumOp->removeAttr("reduction_axes");
    sumOp->removeAttr("reduction_type");
    sum = sumOp->getResult(0);
  } else {
    // Max/min recombine through a comparison and select: the scalar PIL
    // layer has no max/min callee, but gt/lt comparisons and the select
    // conditional-expression lowering are both first-class (stage two
    // maps them onto the `gt`/`lt` and `select` pseudo-callees, which
    // the translator spells as infix comparisons and Python
    // conditionals).
    bool floatKind = isa<arith::MaximumFOp, arith::MinimumFOp>(item.bin);
    Value cond;
    if (floatKind) {
      auto pred = isa<arith::MaximumFOp>(item.bin) ? arith::CmpFPredicate::OGT : arith::CmpFPredicate::OLT;
      cond = builder.create<arith::CmpFOp>(loc, pred, scratchSum, tailSum);
    } else {
      auto pred = isa<arith::MaxSIOp>(item.bin)  ? arith::CmpIPredicate::sgt
                  : isa<arith::MaxUIOp>(item.bin) ? arith::CmpIPredicate::ugt
                  : isa<arith::MinSIOp>(item.bin) ? arith::CmpIPredicate::slt
                                                  : arith::CmpIPredicate::ult;
      cond = builder.create<arith::CmpIOp>(loc, pred, scratchSum, tailSum);
    }
    sum = builder.create<arith::SelectOp>(loc, cond, scratchSum, tailSum);
  }
  item.red->getResult(0).replaceAllUsesWith(sum);
  item.red.erase();
  if (item.ins->use_empty()) {
    item.ins.erase();
  }
  if (item.bin->use_empty()) {
    item.bin->erase();
  }
  if (item.ext->use_empty()) {
    item.ext->erase();
  }
}

/// Folds the streaming-reduction tail window the linalg vectorizer stamps
/// over a scratch tile: `reduction(insert_slice(bin(tail, extract_slice(
/// scratch, win)), scratch, win))` becomes `reduction(scratch) +
/// reduction(tail)`. The sum is linear, so the window's read-modify-write
/// is redundant when the only foldable shape is an add whose other operand
/// is exactly the window extracted back out of the same scratch tile; the
/// insert_slice itself then has no legal pypto lowering (the insert-slice
/// pattern only feeds transfer_write/reshape users), which is what this
/// peephole removes. Floating-point reassociation stays within the
/// reduction_type="x" contract the vectorizer already stamps on the add.
static void foldInsertSliceIntoReductions(func::FuncOp funcOp) {
  SmallVector<InsertSliceFoldCandidate> work;
  funcOp.walk([&](npuvector::ReductionOp red) {
    if (auto candidate = matchInsertSliceFold(red)) {
      work.push_back(*candidate);
    }
  });
  for (InsertSliceFoldCandidate &item : work) {
    applyInsertSliceFold(item);
  }
}

/// Lifts a scalar-condition select onto the tile domain before the
/// conversion: `select(cmpf(scalar, scalar), scalarBranch, tile)` cannot
/// legalize -- `pl.tile.select` demands a packed predicate mask and a tile
/// branch geometry -- so the comparison's lhs is broadcast onto the
/// select's (static) result shape, turning the cmp into a tile comparison
/// whose mask the select then consumes natively. The broadcast of a
/// non-constant scalar materializes as a carrier tile; the constant rhs
/// rides the tile-scalar `pl.tile.cmps` path, and a constant scalar branch
/// is the literal form `pl.tile.select` already accepts.
static void liftScalarCmpSelects(func::FuncOp funcOp) {
  struct LiftCandidate {
    npuvector::SelectOp select;
    Operation *cmp;
  };
  SmallVector<LiftCandidate> work;
  funcOp.walk([&](npuvector::SelectOp sel) {
    if (!isa<IntegerType>(sel.getCondition().getType())) {
      return;  // a tile condition needs no lift
    }
    Operation *cmp = sel.getCondition().getDefiningOp();
    if (!isa<npuvector::CmpFOp, npuvector::CmpIOp>(cmp)) {
      return;  // an opaque scalar condition has no tile spelling
    }
    if (llvm::any_of(cmp->getOperands(), [](Value operand) {
          return isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(operand.getType());
        })) {
      return;
    }
    auto shaped = dyn_cast<ShapedType>(sel->getResult(0).getType());
    if (!shaped || !shaped.hasStaticShape() || shaped.getRank() == 0) {
      return;
    }
    // `pl.tile.cmps` takes the rhs as a literal, and `pl.tile.select`
    // materializes a scalar branch through tile.full -- both need compile
    // time constants. The comparison's lhs rides a broadcast (carrier
    // tile plus add), so any scalar value works there.
    for (Value operand : cmp->getOperands()) {
      if (operand == cmp->getOperand(0) ||
          isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(operand.getType())) {
        continue;
      }
      if (!operand.getDefiningOp<arith::ConstantOp>()) {
        return;
      }
    }
    for (Value operand : sel->getOperands()) {
      if (operand == sel.getCondition() ||
          isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(operand.getType())) {
        continue;
      }
      if (!operand.getDefiningOp<arith::ConstantOp>()) {
        return;
      }
    }
    work.push_back({sel, cmp});
  });
  for (LiftCandidate &item : work) {
    OpBuilder builder(item.select);
    Location loc = item.select.getLoc();
    auto shaped = cast<ShapedType>(item.select->getResult(0).getType());
    SmallVector<Value> dims;
    std::transform(shaped.getShape().begin(), shaped.getShape().end(), std::back_inserter(dims),
                   [&](int64_t dim) { return builder.create<arith::ConstantIndexOp>(loc, dim); });
    Value lifted = builder
                       .create<npuvector::BroadcastOp>(loc, shaped, item.cmp->getOperand(0), ValueRange(dims),
                                                       ValueRange(dims))
                       .getResult();
    Operation *liftedCmp = builder.clone(*item.cmp);
    liftedCmp->setOperand(0, lifted);
    // The source comparison spelled a scalar chain, so its result was the
    // scalar i1. With the lhs riding the broadcast tile the mask must
    // re-type onto the tile domain - a size-1 tile of i1 sharing the
    // select's geometry - otherwise CmpToPypto forwards the scalar i1
    // into pypto.cmp and the select's condition never becomes a tile.
    Type maskType = npuvector::NPUVectorType::get(shaped.getShape(), builder.getI1Type());
    liftedCmp->getResult(0).setType(maskType);
    Operation *liftedSel = builder.clone(*item.select);
    liftedSel->setOperand(0, liftedCmp->getResult(0));
    item.select->getResult(0).replaceAllUsesWith(liftedSel->getResult(0));
    item.select.erase();
    if (item.cmp->use_empty()) {
      item.cmp->erase();
    }
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
      ValueRange indices =
        TypeSwitch<Operation *, ValueRange>(op)
          .Case<npuvector::TransferReadOp>([](npuvector::TransferReadOp read) { return read.getIndices(); })
          .Case<vector::TransferReadOp>([](vector::TransferReadOp read) { return read.getIndices(); });
      builder.setInsertionPoint(op);
      Value read = builder.create<pypto::ReadOp>(op->getLoc(), shaped.getElementType(), source, indices).getResult();
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
          ((ns == "arith" || ns == "math") || isa<npuvector::CmpFOp, npuvector::CmpIOp, npuvector::SelectOp>(op));
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

    // Pre-conversion peepholes over the npuvector IR: fold the
    // streaming-reduction insert-slice windows onto sum linearity, and
    // lift scalar-condition selects onto the tile domain, both before the
    // type converter and target freeze the value types.
    foldInsertSliceIntoReductions(funcOp);
    liftScalarCmpSelects(funcOp);

    VectorToPyptoTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    // Patterns create both the target operations and their physical Tile types.
    target.addDynamicallyLegalDialect<pypto::PyptoDialect>([](Operation *op) {
      return llvm::none_of(op->getResultTypes(), [](Type type) { return isa<npuvector::NPUVectorType>(type); });
    });
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
                                 arith::SelectOp, arith::FloorDivSIOp, arith::RemSIOp, arith::RemUIOp, arith::NegFOp>(
      [](Operation *op) {
        return !isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(op->getOperand(0).getType());
      });
    // Scalar float math keeps the math op the same way: `pl.sqrt`/`pl.rsqrt`
    // are Tensor/Tile-only APIs, so scalar square roots (e.g. the batch-norm
    // scale computation) wait for stage two, which maps them onto
    // pseudo-callees the translator lowers through a small tile detour
    // (the scalar `** 0.5` power form is numerically wrong on the backend).
    target.addDynamicallyLegalOp<math::SqrtOp, math::RsqrtOp, math::ExpOp, math::LogOp>([](Operation *op) {
      return !isa<VectorType, npuvector::NPUVectorType, pypto::TileType>(op->getOperand(0).getType());
    });
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
    // Convert function parameters and calls at the dialect boundary.
    // Tensor views retain memref types until the downstream annotation pass;
    // returns were emptied by the output-parameter rewrite.
    target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addLegalOp<func::ReturnOp>();
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) { return typeConverter.isLegal(op); });

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
static void populateFloatArithPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Constants.
  patterns.add<ArithConstantPattern>(ctx);

  // Elementwise float arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddFOp, pypto::AddOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubFOp, pypto::SubOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulFOp, pypto::MulOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivFOp, pypto::DivOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::RemFOp, pypto::FmodOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::MaximumFOp, pypto::MaximumOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinimumFOp, pypto::MinimumOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::NegFOp, pypto::NegOp>>(ctx);
}

/// Elementwise integer arithmetic.
static void populateIntArithPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Elementwise integer arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddIOp, pypto::AddOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubIOp, pypto::SubOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulIOp, pypto::MulOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivSIOp, pypto::DivOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivUIOp, pypto::DivOp>>(ctx);
  patterns.add<TileRemToPyptoPattern<arith::RemSIOp>>(ctx);
  patterns.add<TileRemToPyptoPattern<arith::RemUIOp>>(ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MaxSIOp, pypto::MaxOp, pypto::MaximumOp>>(ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MinSIOp, pypto::MinOp, pypto::MinimumOp>>(ctx);
  // Floor division stays arith.floordivsi here: the translator prints the
  // `floordivsi` pseudo-callee it lowers to as an infix `(a // b)`
  // expression, and convert-pypto-to-emitpy introduces that call.
  patterns.add<OpToPyptoOpPattern<arith::MaxUIOp, pypto::MaximumOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinUIOp, pypto::MinimumOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::AndIOp, pypto::AndOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::OrIOp, pypto::OrOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::XOrIOp, pypto::XorOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShLIOp, pypto::ShlOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRSIOp, pypto::ShrOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRUIOp, pypto::ShrOp>>(ctx);
}

/// Comparisons and select.
static void populateCmpSelectPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Comparisons and select.
  patterns.add<CmpToPyptoPattern<arith::CmpIOp>>(ctx);
  patterns.add<CmpToPyptoPattern<arith::CmpFOp>>(ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpIOp>>(ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpFOp>>(ctx);
  patterns.add<SelectPattern<arith::SelectOp>>(ctx);
  patterns.add<SelectPattern<npuvector::SelectOp>>(ctx);
}

/// Numeric casts (pl.cast) and bitcasts (pl.reinterpret_view).
static void populateCastPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Numeric casts -> pl.cast; bitcasts -> pl.reinterpret_view.
  patterns.add<CastToPyptoPattern<arith::ExtFOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::TruncFOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::ExtSIOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::ExtUIOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::TruncIOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::SIToFPOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::UIToFPOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::FPToSIOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::FPToUIOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastOp>>(ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastUIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtFOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncFOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtSIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtUIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::SIToFPOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::UIToFPOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToSIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToUIOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastOp>>(ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastUIOp>>(ctx);
  patterns.add<BitcastToPyptoPattern<arith::BitcastOp>>(ctx);
  patterns.add<BitcastToPyptoPattern<npuvector::BitcastOp>>(ctx);
}

/// Rewrites `math.round`/`math.floor`/`math.ceil` on a tile into the PyPTO
/// round-trip cast chain: `pl.cast` rounds float-to-int with the named
/// rounding mode, and the widening cast back restores the source dtype.
/// PyPTO has no first-class rounding op.
template <typename OpTy>
class RoundingCastPattern : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  RoundingCastPattern(MLIRContext *ctx, StringRef mode) : OpConversionPattern<OpTy>(ctx), mode(mode.str()) {}

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
    auto physical = cast<ShapedType>(adaptor.getOperands()[0].getType());
    Type intTile = pypto::TileType::get(rewriter.getContext(), rewriter.getIntegerType(32), physical.getShape());
    Value asInt = rewriter.create<pypto::CastOp>(loc, pyptoType(intTile), adaptor.getOperands()[0]);
    asInt.getDefiningOp<pypto::CastOp>()->setAttr("mode", rewriter.getStringAttr(mode));
    rewriter.replaceOpWithNewOp<pypto::CastOp>(op, pyptoResultTypes(op, adaptor.getOperands()), asInt);
    return success();
  }

 private:
  std::string mode;
};

/// Math dialect elementwise functions.
static void populateMathPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // Math dialect elementwise functions with PyPTO counterparts.
  patterns.add<OpToPyptoOpPattern<math::AbsFOp, pypto::AbsOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::AbsIOp, pypto::AbsOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::ExpOp, pypto::ExpOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::LogOp, pypto::LogOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::SqrtOp, pypto::SqrtOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::RsqrtOp, pypto::RsqrtOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::SinOp, pypto::SinOp>>(ctx);
  patterns.add<OpToPyptoOpPattern<math::CosOp, pypto::CosOp>>(ctx);
  patterns.add<RoundingCastPattern<math::RoundOp>>(ctx, "round");
  patterns.add<RoundingCastPattern<math::FloorOp>>(ctx, "floor");
  patterns.add<RoundingCastPattern<math::CeilOp>>(ctx, "ceil");
}

/// Tile ops, memref management, and control flow.
static void populateTileAndControlFlowPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // npuvector / vector tile ops.
  patterns.add<TransferReadPattern<npuvector::TransferReadOp>>(ctx);
  patterns.add<TransferReadPattern<vector::TransferReadOp>>(ctx);
  patterns.add<TransferWritePattern<npuvector::TransferWriteOp>>(ctx);
  patterns.add<TransferWritePattern<vector::TransferWriteOp>>(ctx);
  patterns.add<ExtractSlicePattern>(ctx);
  patterns.add<InsertSlicePattern>(ctx);
  patterns.add<ReductionPattern<npuvector::ReductionOp>>(ctx);
  patterns.add<ReductionPattern<vector::ReductionOp>>(ctx);
  // Comparison-mask lift before both broadcast fusions: a rank-1 `i1`
  // comparison lifted to rank-2 rebuilds `pypto.cmp` at the broadcast's
  // geometry (pl.cmp never broadcasts and pl.tile.select demands a
  // full-geometry condition). The benefit outranks the tile-broadcast
  // fusion so an i1 source cannot stray onto the arithmetic expand
  // family, and the always-matching fallback.
  patterns.add<BroadcastCmpMaskPattern>(ctx, PatternBenefit(3));
  // Tile-broadcast fusion before the scalar-broadcast fallback: a rank-1
  // tile lifted to rank-2 by `dimension = [0]/[1]` and consumed by arith
  // add/sub/mul/div folds into the runtime-native row/col_expand family,
  // fusing every consumer. The benefit outranks BroadcastPattern so the
  // always-matching fallback cannot starve the fusion.
  patterns.add<TileBroadcastFusionPattern>(ctx, PatternBenefit(2));
  // Scalar-broadcast fusion before the scalar-broadcast fallback: a
  // scalar consumed by comparisons and select branches rides through as
  // the tile-scalar `pl.tile.cmps` / `pl.tile.select` forms. The fusion
  // carries a higher benefit than BroadcastPattern: with equal benefits
  // the pattern order is unspecified, and the always-matching fallback
  // would otherwise starve the fusion.
  patterns.add<ScalarBroadcastFusionPattern>(ctx, PatternBenefit(2));
  patterns.add<BroadcastPattern<npuvector::BroadcastOp>>(ctx);
  patterns.add<BroadcastPattern<vector::BroadcastOp>>(ctx);
  patterns.add<TransposePattern<npuvector::TransposeOp>>(ctx);
  patterns.add<TransposePattern<vector::TransposeOp>>(ctx);

  // memref management ops.
  patterns.add<CollapseShapePattern>(ctx);
  patterns.add<ExpandShapePattern>(ctx);
  patterns.add<MemrefTransposePattern>(ctx);
  patterns.add<ReshapePattern>(ctx);
  patterns.add<SubviewPattern>(ctx);
  patterns.add<ReinterpretCastPattern>(ctx);
  patterns.add<DeallocPattern>(ctx);

  // Control flow becomes pypto: scf.parallel/scf.forall map onto `pypto.spmd`
  // multi-core launches (the induction variable is the executing core's block
  // index) and scf.for onto the `pypto.range` per-core tile loop. Stage two
  // then only maps these pypto ops onto their emitpy spellings.
  patterns.add<ScfParallelPattern>(ctx);
  patterns.add<ScfForallPattern>(ctx);
  patterns.add<ScfForPattern>(ctx);
}

}  // namespace

void populateVectorToPyptoConversionPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  populateFloatArithPatterns(patterns);
  populateIntArithPatterns(patterns);
  populateCmpSelectPatterns(patterns);
  populateCastPatterns(patterns);
  populateMathPatterns(patterns);
  populateTileAndControlFlowPatterns(patterns);

  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
}

}  // namespace vector_to_pypto

std::unique_ptr<Pass> createVectorToPyptoConversionPass() { return std::make_unique<ConvertVectorToPyptoPass>(); }

}  // namespace mlir
